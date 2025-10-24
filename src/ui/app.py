# src/ui/app.py
import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
API_BASE = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

def init_session_state():
    """Initialize session state variables"""
    if 'analysis_requests' not in st.session_state:
        st.session_state.analysis_requests = []
    if 'current_analysis_id' not in st.session_state:
        st.session_state.current_analysis_id = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🤖 Autonomous Due Diligence Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    .completed-analysis {
        border-left: 5px solid #28a745;
    }
    .running-analysis {
        border-left: 5px solid #ffc107;
    }
    .failed-analysis {
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Autonomous Due Diligence Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    **AI-powered comprehensive company analysis for investment decisions**
    
    *Multi-agent system analyzing financials, legal compliance, and market position*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Health Check
        if st.button("🩺 Check API Health"):
            check_api_health()
        
        st.markdown("---")
        st.header("📈 Recent Analyses")
        display_recent_analyses()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🚀 New Analysis", "📊 Results", "📋 Analysis History"])
    
    with tab1:
        new_analysis_tab()
    
    with tab2:
        results_tab()
    
    with tab3:
        history_tab()

def new_analysis_tab():
    """Tab for starting new analyses"""
    st.header("Start New Due Diligence Analysis")
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_ticker = st.text_input(
                "Company Ticker Symbol",
                placeholder="AAPL, MSFT, TSLA...",
                help="Enter the stock ticker symbol in uppercase",
                value="AAPL"
            ).upper()
            
            analysis_type = st.selectbox(
                "Analysis Type",
                options=["comprehensive", "financial", "legal", "market"],
                help="Choose the type of analysis to perform",
                index=0
            )
        
        with col2:
            priority = st.selectbox(
                "Priority Level",
                options=["low", "normal", "high"],
                help="Analysis priority affects processing speed",
                index=1
            )
            
            user_id = st.text_input(
                "User ID (Optional)",
                placeholder="Your identifier",
                help="Optional identifier for tracking"
            )
        
        # Questions input
        st.subheader("Specific Analysis Questions")
        questions = st.text_area(
            "Questions to Answer",
            placeholder="What are the main financial risks?\nHow is revenue trending?\nWhat is the competitive landscape?",
            help="Enter specific questions you want answered (one per line)",
            height=120
        )
        
        # Form submission
        submitted = st.form_submit_button(
            "🚀 Start Due Diligence Analysis",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if not company_ticker:
                st.error("❌ Please enter a company ticker symbol")
                return
            
            start_analysis(
                company_ticker=company_ticker,
                analysis_type=analysis_type,
                questions=questions,
                user_id=user_id,
                priority=priority
            )

def results_tab():
    """Tab for viewing analysis results"""
    st.header("Analysis Results")
    
    if not st.session_state.current_analysis_id:
        st.info("👆 Start an analysis first to see results here")
        return
    
    # Get current analysis status
    analysis_id = st.session_state.current_analysis_id
    status = get_analysis_status(analysis_id)
    
    if not status:
        st.error("❌ Could not retrieve analysis status")
        return
    
    # Display progress and status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", status['status'].title())
    
    with col2:
        st.metric("Progress", f"{status['progress'] * 100:.1f}%")
    
    with col3:
        if status['estimated_completion']:
            st.metric("Estimated Completion", status['estimated_completion'])
    
    # Progress bar
    st.progress(status['progress'])
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh results", 
        value=st.session_state.auto_refresh,
        help="Automatically refresh results every 5 seconds"
    )
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    
    # Display results if completed
    if status['status'] == 'completed' and status.get('results'):
        display_analysis_results(status['results'])
    elif status['status'] == 'failed':
        st.error(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
    elif status['status'] in ['initialized', 'running']:
        st.info(f"🔄 Analysis in progress... Current step: {status.get('current_step', 'Unknown')}")
        
        # Manual refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()

def history_tab():
    """Tab for viewing analysis history"""
    st.header("Analysis History")
    
    analyses = list_analyses()
    
    if not analyses:
        st.info("No analyses found. Start your first analysis above!")
        return
    
    # Display analyses in a table
    df = pd.DataFrame(analyses)
    st.dataframe(df, use_container_width=True)
    
    # Option to view details
    selected_analysis = st.selectbox(
        "Select analysis to view details",
        options=analyses,
        format_func=lambda x: f"{x['request_id']} - {x['company']} - {x['status']}"
    )
    
    if selected_analysis and st.button("View Detailed Results"):
        st.session_state.current_analysis_id = selected_analysis['request_id']
        st.rerun()

def display_analysis_results(results: dict):
    """Display comprehensive analysis results"""
    st.success("✅ Analysis Completed Successfully!")
    
    # Executive Summary
    with st.expander("📋 Executive Summary", expanded=True):
        if results.get('synthesis_report'):
            synthesis = results['synthesis_report']
            st.subheader("Investment Recommendation")
            st.metric("Overall Recommendation", synthesis.get('recommendation', 'N/A'))
            st.metric("Overall Risk", synthesis.get('risk_assessment', {}).get('overall_risk', 'N/A'))
            
            st.subheader("Key Findings")
            if synthesis.get('key_findings'):
                for category, findings in synthesis['key_findings'].items():
                    st.write(f"**{category.title()}:**")
                    if isinstance(findings, dict):
                        for k, v in findings.items():
                            st.write(f"  - {k}: {v}")
                    else:
                        st.write(f"  - {findings}")
    
    # Financial Analysis
    if results.get('financial_analysis'):
        with st.expander("📊 Financial Analysis", expanded=True):
            financial = results['financial_analysis']
            
            if financial.get('status') == 'completed':
                # Financial Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if financial.get('key_metrics', {}).get('profitability'):
                        st.metric("Profitability", financial['key_metrics']['profitability'])
                
                with col2:
                    if financial.get('key_metrics', {}).get('liquidity'):
                        st.metric("Liquidity", financial['key_metrics']['liquidity'])
                
                with col3:
                    if financial.get('key_metrics', {}).get('efficiency'):
                        st.metric("Efficiency", financial['key_metrics']['efficiency'])
                
                with col4:
                    if financial.get('risk_factors'):
                        st.metric("Risk Factors", len(financial['risk_factors']))
                
                # Detailed Analysis
                if financial.get('analysis'):
                    st.subheader("Detailed Analysis")
                    st.write(financial['analysis'])
                
                # Risk Factors
                if financial.get('risk_factors'):
                    st.subheader("Financial Risk Factors")
                    for risk in financial['risk_factors']:
                        st.write(f"⚠️ {risk}")
            
            else:
                st.warning("Financial analysis incomplete or failed")
    
    # Legal Analysis
    if results.get('legal_analysis'):
        with st.expander("⚖️ Legal Analysis"):
            legal = results['legal_analysis']
            if legal.get('status') == 'completed':
                st.metric("Compliance Status", legal.get('compliance_status', 'Unknown'))
                
                if legal.get('risk_factors'):
                    st.subheader("Legal Risk Factors")
                    for risk in legal['risk_factors']:
                        st.write(f"⚖️ {risk}")
            else:
                st.warning("Legal analysis incomplete or failed")
    
    # Market Analysis
    if results.get('market_analysis'):
        with st.expander("📈 Market Analysis"):
            market = results['market_analysis']
            if market.get('status') == 'completed':
                st.metric("Market Position", market.get('market_position', 'Unknown'))
                
                if market.get('competitors'):
                    st.subheader("Key Competitors")
                    st.write(", ".join(market['competitors']))
            else:
                st.warning("Market analysis incomplete or failed")
    
    # Download Results
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=json.dumps(results, indent=2),
        file_name=f"due_diligence_{results.get('company', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def start_analysis(company_ticker: str, analysis_type: str, questions: str, user_id: str, priority: str):
    """Start a new analysis"""
    try:
        # Prepare request
        request_data = {
            "company_ticker": company_ticker,
            "analysis_type": analysis_type,
            "questions": [q.strip() for q in questions.split('\n') if q.strip()] if questions else [],
            "user_id": user_id if user_id else None,
            "priority": priority
        }
        
        # Call API
        with st.spinner("🚀 Starting analysis..."):
            response = requests.post(f"{API_BASE}/analyze", json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis_id = result['request_id']
            
            # Store in session state
            st.session_state.current_analysis_id = analysis_id
            st.session_state.analysis_requests.append({
                "request_id": analysis_id,
                "company": company_ticker,
                "type": analysis_type,
                "start_time": datetime.now().isoformat(),
                "status": "initialized"
            })
            
            st.success(f"✅ Analysis started! Request ID: {analysis_id}")
            st.info(f"📊 Analysis will take approximately {result['estimated_completion']}")
            
            # Switch to results tab
            st.rerun()
            
        else:
            st.error(f"❌ Failed to start analysis: {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API connection error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

def get_analysis_status(analysis_id: str) -> dict:
    """Get status of a specific analysis"""
    try:
        response = requests.get(f"{API_BASE}/analysis/{analysis_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Failed to get analysis status: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API connection error: {str(e)}")
        return None

def list_analyses() -> list:
    """List all analyses"""
    try:
        response = requests.get(f"{API_BASE}/analyses", timeout=10)
        if response.status_code == 200:
            return response.json()['analyses']
        return []
    except:
        return []

def check_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            if health['status'] == 'healthy':
                st.success("✅ API is healthy and operational")
            else:
                st.warning("⚠️ API is degraded")
            
            # Display component status
            for component, status in health['components'].items():
                if isinstance(status, dict):
                    status_str = "✅ Healthy" if status.get('status') == 'healthy' else "❌ Issues"
                else:
                    status_str = "✅ Healthy" if status else "❌ Issues"
                st.write(f"- {component}: {status_str}")
        else:
            st.error("❌ API health check failed")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")

def display_recent_analyses():
    """Display recent analyses in sidebar"""
    analyses = list_analyses()[:5]  # Last 5 analyses
    
    if not analyses:
        st.write("No recent analyses")
        return
    
    for analysis in analyses:
        status_emoji = {
            'completed': '✅',
            'running': '🔄', 
            'failed': '❌',
            'initialized': '⏳'
        }.get(analysis['status'], '❓')
        
        st.write(f"{status_emoji} {analysis['company']} - {analysis['status']}")
        
        if st.button(f"View {analysis['request_id'][-8:]}", key=analysis['request_id']):
            st.session_state.current_analysis_id = analysis['request_id']
            st.rerun()

if __name__ == "__main__":
    main()