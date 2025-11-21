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
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #1e1e1e;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
    }
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
            "additional_context": context
        }
        try:
            response = self.session.post(f"{self.base_url}/analyze", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None
        except requests.RequestException as e:
            st.error(f"Connection Error: {e}")
            return None

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        try:
            response = self.session.get(f"{self.base_url}/analysis/{session_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.RequestException:
            return None

# Initialize Client
api_client = ApiClient(API_BASE_URL)

def render_sidebar():
    """Render the sidebar navigation and info"""
    st.sidebar.title("🕵️‍♂️ Due Diligence")
    st.sidebar.caption("Autonomous Agentic System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", ["New Analysis", "System Status"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 History")
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    if st.session_state.history:
        for item in st.session_state.history[-5:]: # Show last 5
            st.sidebar.text(f"{item['ticker']} - {item['timestamp']}")
    else:
        st.sidebar.caption("No recent analyses")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.text_input("API Endpoint", value=API_BASE_URL, disabled=True)
    
    return page

def render_system_status():
    """Render the system status dashboard"""
    st.title("🖥️ System Status")
    
    with st.spinner("Checking system health..."):
        is_healthy, health_data = api_client.check_health()
    
    if is_healthy:
        st.success("✅ System is Online and Operational")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Uptime", f"{health_data.get('uptime', 0)/3600:.1f}h", delta="Running")
        with col2:
            st.metric("Active Sessions", health_data.get('active_sessions', 0))
        with col3:
            st.metric("Version", health_data.get('version', '1.0.0'))
        with col4:
            status = health_data.get('status', 'Unknown').upper()
            st.metric("Global Status", status, delta_color="normal" if status=="HEALTHY" else "inverse")
            
        # Component Health Grid
        st.subheader("Component Health")
        components = health_data.get('components', {})
        
        cols = st.columns(3)
        for i, (comp, status) in enumerate(components.items()):
            with cols[i % 3]:
                status_color = "🟢" if status == "healthy" else "🔴"
                with st.container(border=True):
                    st.markdown(f"**{comp.replace('_', ' ').title()}**")
                    st.markdown(f"{status_color} {status.upper()}")
    else:
        st.error("❌ System is Offline")
        st.warning(f"Cannot connect to backend API at `{API_BASE_URL}`")
        st.info("Please ensure the FastAPI backend is running: `python src/api/main.py`")

def render_new_analysis():
    """Render the new analysis workflow"""
    st.title("🚀 New Analysis")
    st.markdown("Start a comprehensive autonomous due diligence analysis.")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            ticker = st.text_input(
                "Company Ticker", 
                placeholder="e.g., AAPL, MSFT",
                help="Enter the stock ticker symbol"
            ).upper()
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["comprehensive", "financial", "legal", "market"],
                help="Select the depth and focus of the analysis"
            )
            
        with col2:
            priority = st.select_slider(
                "Priority Level", 
                options=["low", "normal", "high"], 
                value="normal",
                help="Higher priority tasks are processed first"
            )
            
            context = st.text_area(
                "Additional Context", 
                placeholder="Specific areas to focus on (e.g., 'Check for recent litigation' or 'Analyze Q3 revenue growth')",
                height=100
            )
            
        start_btn = st.button("Initiate Analysis", type="primary")

    if start_btn and ticker:
        run_analysis_flow(ticker, analysis_type, priority, context)

def run_analysis_flow(ticker: str, analysis_type: str, priority: str, context: str):
    """Handle the analysis execution flow"""
    with st.status(f"Initializing analysis for {ticker}...", expanded=True) as status:
        st.write("📡 Connecting to agent swarm...")
        result = api_client.start_analysis(ticker, analysis_type, priority, context)
        
        if result:
            session_id = result['session_id']
            st.write(f"✅ Session created: `{session_id}`")
            st.write("🤖 Agents activated. Starting data collection...")
            
            # Add to history
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                'ticker': ticker,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            # Progress Loop
            progress_bar = st.progress(0)
            
            while True:
                session_status = api_client.get_status(session_id)
                if not session_status:
                    status.update(label="⚠️ Connection lost", state="error")
                    st.error("Lost connection to session.")
                    break
                
                current_state = session_status.get('status')
                progress = session_status.get('progress', 0)
                
                # Update UI
                progress_bar.progress(progress)
                
                # Log updates (simulated based on state for better UX)
                if current_state == "queued":
                    st.write("⏳ Analysis queued...")
                elif current_state == "processing":
                    pass 
                
                if current_state == "completed":
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    progress_bar.progress(100)
                    render_results(session_status.get('result', {}), ticker)
                    break
                
                if current_state == "failed":
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {session_status.get('error')}")
                    break
                
                time.sleep(2)
        else:
            status.update(label="❌ Initialization Failed", state="error")

def render_financial_charts(data: Dict[str, Any]):
    """Render financial charts using Plotly"""
    # Mock data for visualization if real data is missing or unstructured
    # In a real app, you'd parse the actual financial data
    
    st.subheader("Financial Performance")
    
    # Mock Data
    years = ['2020', '2021', '2022', '2023']
    revenue = [100, 120, 150, 180]
    net_income = [20, 25, 35, 45]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=revenue, name='Revenue', marker_color='#4facfe'))
    fig.add_trace(go.Scatter(x=years, y=net_income, name='Net Income', line=dict(color='#00f2fe', width=4)))
    
    fig.update_layout(
        title='Revenue vs Net Income (Mock Data)',
        xaxis_title='Year',
        yaxis_title='Amount ($B)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_structured_findings(findings: Dict[str, Any]):
    """Render findings in a structured, readable format"""
    
    # Financial Analysis
    if 'financial_analysis' in findings:
        with st.expander("💰 Financial Analysis", expanded=True):
            st.markdown(findings['financial_analysis'])
            
    # Market Analysis
    if 'market_analysis' in findings:
        with st.expander("🌍 Market Analysis", expanded=True):
            st.markdown(findings['market_analysis'])
            
    # Legal Analysis
    if 'legal_analysis' in findings:
        with st.expander("⚖️ Legal Analysis", expanded=True):
            st.markdown(findings['legal_analysis'])

def generate_report_markdown(result: Dict[str, Any], ticker: str) -> str:
    """Generate a markdown report string"""
    report = f"# Due Diligence Report: {ticker}\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    if 'summary' in result:
        report += f"## Executive Summary\n{result['summary']}\n\n"
        
    if 'financial_analysis' in result:
        report += f"## Financial Analysis\n{result['financial_analysis']}\n\n"
        
    if 'market_analysis' in result:
        report += f"## Market Analysis\n{result['market_analysis']}\n\n"
        
    if 'legal_analysis' in result:
        report += f"## Legal Analysis\n{result['legal_analysis']}\n\n"
        
    return report

def render_results(result: Dict[str, Any], ticker: str):
    """Render the analysis results"""
    st.divider()
    st.header("📊 Analysis Report")
    
    # Download Button
    report_md = generate_report_markdown(result, ticker)
    st.download_button(
        label="📥 Download Report",
        data=report_md,
        file_name=f"{ticker}_due_diligence_report.md",
        mime="text/markdown"
    )
    
    # Summary Section
    if 'summary' in result:
        st.markdown("### Executive Summary")
        st.info(result['summary'])
    
    # Detailed Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Detailed Findings", "📈 Financial Data", "⚖️ Legal Risks"])
    
    with tab1:
        st.markdown("#### Agent Findings")
        render_structured_findings(result)
        
    with tab2:
        st.markdown("#### Financial Metrics")
        render_financial_charts(result)
        
    with tab3:
        st.markdown("#### Legal Risk Assessment")
        # Placeholder for legal risk visualization
        if 'legal_analysis' in result:
             st.markdown(result['legal_analysis'])
        else:
            st.warning("Legal risk assessment data would appear here.")

# Main App Entry Point
def main():
    page = render_sidebar()
    
    if page == "System Status":
        render_system_status()
    elif page == "New Analysis":
        render_new_analysis()

if __name__ == "__main__":
    main()
