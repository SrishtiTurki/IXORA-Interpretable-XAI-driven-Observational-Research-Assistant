"""
app.py - IXORA Research Assistant - OPTIMIZED VERSION
Shows: Analytics Method (LIME/SHAP), AI Trace, Validation Scores prominently
"""

import streamlit as st
import requests
from requests.exceptions import Timeout, ConnectionError
import uuid
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost").rstrip("/")
DEFAULT_PORT = 8000
DOMAIN_PORTS = {
    "biomed": int(os.getenv("BIOMED_PORT", DEFAULT_PORT)),
    "cs": int(os.getenv("CS_PORT", DEFAULT_PORT)),
    "general": int(os.getenv("GENERAL_PORT", DEFAULT_PORT))
}

st.set_page_config(
    page_title="🧪 IXORA - Multi-Agent Research Assistant",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #667eea;
        padding: 1rem;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle { 
        text-align: center; 
        color: #666; 
        margin-bottom: 2rem; 
        font-size: 1.1rem; 
    }
    .user-message {
        background-color: #e3f2fd; 
        border-left: 4px solid #2196f3;
        padding: 1rem; 
        border-radius: 8px; 
        margin: 1rem 0; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #f8f9fa; 
        border-left: 4px solid #764ba2;
        padding: 1.5rem; 
        border-radius: 10px; 
        margin: 1rem 0; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .param-box { 
        background: #e8f5e9; 
        border-left: 4px solid #4caf50; 
        padding: 0.75rem; 
        border-radius: 6px; 
        margin: 0.5rem 0; 
    }
    .confidence-badge { 
        display: inline-block; 
        padding: 0.3rem 0.8rem; 
        border-radius: 20px; 
        font-weight: bold; 
        font-size: 0.9rem; 
        margin-top: 0.5rem;
    }
    .confidence-high { background: #4caf50; color: white; }
    .confidence-medium { background: #ff9800; color: white; }
    .confidence-low { background: #f44336; color: white; }
    .section-header { 
        font-size: 1.4rem; 
        font-weight: 600; 
        color: #333; 
        margin: 1.5rem 0 1rem; 
        padding-bottom: 0.5rem; 
        border-bottom: 2px solid #667eea; 
    }
    .metric-card { 
        background: white; 
        padding: 1rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.1); 
        text-align: center; 
        margin: 0.5rem 0;
    }
    .trace-step { 
        background: #f5f5f5; 
        border-left: 4px solid #667eea; 
        padding: 1rem; 
        border-radius: 6px; 
        margin: 0.5rem 0;
    }
    .method-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stChatInput > div > div > input {
        border-radius: 10px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "extracted_parameters": {},
        "trace_data": [],
        "validation_scores": {},
        "analytics_result": {},
        "causal_analysis": {},
        "embedding_scores": {},
        "last_response": {},
        "domain": "biomed",
        "arxiv_papers": [],
        "show_analytics": False,
        "analytics_method": "none",
        "optimization_status": "not_started",
        "active_tab": "chat",
        "api_connected": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== HELPER FUNCTIONS ====================
def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"

def format_trace_step(step: Dict) -> Dict:
    """Format a trace step for display"""
    if not isinstance(step, dict):
        return {
            "name": "Unknown Step",
            "time": "N/A",
            "summary": str(step)[:100],
            "details": {}
        }
    
    step_name = step.get("step", "Unknown")
    timestamp = step.get("timestamp", datetime.now().isoformat())
    summary = step.get("summary", "")
    
    # Parse timestamp
    try:
        if "T" in timestamp:
            time_part = timestamp.split("T")[1][:8]
        else:
            time_part = timestamp[:8] if len(timestamp) >= 8 else "N/A"
    except:
        time_part = "N/A"
    
    # Build details excluding redundant fields
    details = {}
    for key, value in step.items():
        if key not in ["step", "timestamp", "summary"]:
            details[key] = value
    
    return {
        "name": step_name.replace("_", " ").title(),
        "time": time_part,
        "summary": summary,
        "details": details
    }

def extract_xml_sections(text: str) -> Dict[str, str]:
    """Extract XML sections from response"""
    sections = {}
    patterns = {
        "enthusiasm": r'<enthusiasm>(.*?)</enthusiasm>',
        "clarify": r'<clarify>(.*?)</clarify>',
        "explanation": r'<explanation>(.*?)</explanation>',
        "hypothesis": r'<hypothesis>(.*?)</hypothesis>',
        "followup": r'<followup>(.*?)</followup>'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    
    return sections

def parse_analytics_result(analytics: Dict) -> Dict:
    """Parse analytics result safely"""
    if not analytics or not isinstance(analytics, dict):
        return {
            "explainability": {},
            "explainability_method": "none",
            "optimization": {},
            "causal": {}
        }
    
    return {
        "explainability": analytics.get("explainability", {}),
        "explainability_method": analytics.get("explainability_method", "none"),
        "optimization": analytics.get("optimization", {}),
        "causal": analytics.get("causal", {})
    }

def test_api_connection(domain: str) -> bool:
    """Test connection to API server"""
    try:
        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(domain, DEFAULT_PORT)}/health"
        response = requests.get(api_url, timeout=5)
        return response.status_code == 200
    except:
        return False

# ==================== DISPLAY FUNCTIONS ====================
def display_analytics_method_badge():
    """Display which analytics method was used - PROMINENTLY"""
    method = st.session_state.analytics_method
    
    if method and method != "none":
        method_descriptions = {
            "shap": "SHAP analyzes global feature importance using game theory",
            "lime": "LIME provides local explanations for individual predictions",
            "fast": "Fast heuristic analysis based on domain rules",
            "mixed": "Mixed methods combining SHAP and LIME"
        }
        
        description = method_descriptions.get(method.lower(), 
                     f"{method.upper()} analysis method used")
        
        st.markdown(f"""
        <div class="highlight-box" style="text-align: center;">
            <h3 style="margin: 0 0 0.5rem 0;">🔬 Analytics Method Used</h3>
            <div class="method-badge">{method.upper()}</div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_validation_summary():
    """Display validation scores PROMINENTLY at top"""
    if not st.session_state.validation_scores and not st.session_state.messages:
        return
    
    # Get confidence from last message
    confidence = 0.7
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if isinstance(last_msg, dict) and "confidence" in last_msg:
            confidence = last_msg["confidence"]
    
    validation_scores = st.session_state.validation_scores
    
    st.markdown("### ✅ Response Validation Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        conf_class = get_confidence_class(confidence)
        color = "#4caf50" if confidence >= 0.8 else "#ff9800" if confidence >= 0.6 else "#f44336"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Overall Confidence</h4>
            <h2 style="color: {color}">{confidence:.0%}</h2>
            <small>{conf_class.title()} Confidence</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        structural = validation_scores.get("structural_completeness", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Structure</h4>
            <h2>{structural:.0%}</h2>
            <small>Format quality</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bertscore = validation_scores.get("bertscore_f1", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>BERTScore</h4>
            <h2>{bertscore:.2f}</h2>
            <small>Semantic similarity</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bleurt = validation_scores.get("bleurt_score", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>BLEURT</h4>
            <h2>{bleurt:.2f}</h2>
            <small>Quality metric</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        has_hypothesis = validation_scores.get("has_hypothesis", 0.0)
        has_followup = validation_scores.get("has_followup", 0.0)
        completeness = (has_hypothesis + has_followup) / 2 if (has_hypothesis + has_followup) > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Completeness</h4>
            <h2>{completeness:.0%}</h2>
            <small>Content check</small>
        </div>
        """, unsafe_allow_html=True)

def display_trace_summary():
    """Display trace summary PROMINENTLY"""
    if not st.session_state.trace_data:
        return
    
    trace_data = st.session_state.trace_data
    total_steps = len(trace_data)
    
    st.markdown("### 🧠 AI Reasoning Trace")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Steps", total_steps)
    
    with col2:
        step_types = set()
        for step in trace_data:
            if isinstance(step, dict):
                step_name = step.get("step", "unknown")
                step_types.add(step_name)
        st.metric("Unique Agents", len(step_types))
    
    with col3:
        params_extracted = 0
        for step in trace_data:
            if isinstance(step, dict) and step.get("step") in ["extractor", "parameter_extraction"]:
                params_extracted = step.get("param_count", step.get("parameters_count", 0))
                break
        st.metric("Parameters Extracted", params_extracted)
    
    # Show key steps in timeline
    st.markdown("#### 📋 Key Steps")
    
    for i, step in enumerate(trace_data):
        if not isinstance(step, dict):
            continue
        
        step_name = step.get("step", "unknown")
        
        # Determine icon
        step_icons = {
            "extract": "🔍", "parameter": "🔍", "query": "🔍",
            "draft": "📝", "write": "📝", "generate": "📝",
            "analytics": "📊", "analyze": "📊", "shap": "📊",
            "hypothesis": "🔬", "experiment": "🔬", "test": "🔬",
            "synthesize": "✨", "summarize": "✨", "final": "✨",
            "validate": "✅", "check": "✅", "verify": "✅",
            "optimize": "⚡", "bayesian": "⚡", "causal": "🔗"
        }
        
        step_icon = "🔹"
        for key, icon in step_icons.items():
            if key in step_name.lower():
                step_icon = icon
                break
        
        formatted = format_trace_step(step)
        
        with st.expander(f"{step_icon} Step {i+1}: {formatted['name']} • {formatted['time']}", expanded=False):
            if formatted["summary"]:
                st.markdown(f"**Summary:** {formatted['summary']}")
            
            # Show important details
            if formatted["details"]:
                important_keys = ["param_count", "parameters_count", "explainability_method", 
                                "method", "model", "success", "confidence", "accuracy", 
                                "parameters", "features", "results"]
                
                for key in important_keys:
                    if key in formatted["details"]:
                        value = formatted["details"][key]
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Full details in JSON
            if formatted["details"]:
                with st.expander("🔍 Full Details", expanded=False):
                    st.json(formatted["details"], expanded=False)

def display_parameters_tab(parameters: Dict):
    """Display extracted parameters"""
    if not parameters:
        st.info("No parameters extracted yet.")
        return
    
    st.markdown(f"### 🔢 Extracted Parameters ({len(parameters)})")
    
    for param_name, param in parameters.items():
        with st.expander(f"📊 {param_name.replace('_', ' ').title()}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                value = param.get("value", "N/A")
                unit = param.get("unit", "")
                conf = param.get("confidence", 0.8)
                
                st.markdown(f"**Value:** `{value}` {unit}")
                st.markdown(f"**Confidence:** {conf:.0%}")
                
                conf_class = get_confidence_class(conf)
                st.markdown(f'<span class="confidence-badge confidence-{conf_class}">{conf_class.title()}</span>', 
                          unsafe_allow_html=True)
            
            with col2:
                raw = param.get("raw_text", "")
                method = param.get("method", "unknown")
                
                st.markdown(f"**Extraction Method:** {method}")
                if raw:
                    with st.expander("📝 Original Text", expanded=False):
                        st.text(raw)
                
                if isinstance(value, list) and len(value) == 2:
                    st.success("📊 Range parameter (good for optimization)")
                elif isinstance(value, (int, float)):
                    st.info("📈 Numerical parameter")

def display_explainability_tab(analytics_result: Dict):
    """Display explainability results"""
    if not analytics_result:
        st.info("No explainability analysis available.")
        return
    
    explainability = analytics_result.get("explainability", {})
    method = analytics_result.get("explainability_method", "none")
    
    # Show which method
    if method and method != "none":
        st.success(f"✅ **Primary Method:** {method.upper()}")
    
    # SHAP results
    if "shap" in explainability:
        shap_data = explainability["shap"]
        
        with st.expander("📊 SHAP Analysis", expanded=True):
            if shap_data.get("success"):
                importance = shap_data.get("importance", {})
                
                if importance:
                    imp_df = pd.DataFrame({
                        "Feature": list(importance.keys()),
                        "Importance": list(importance.values())
                    }).sort_values("Importance", ascending=False).head(10)
                    
                    fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                               color="Importance", color_continuous_scale="Viridis",
                               title="Top 10 Feature Importances (SHAP)")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Top Feature", imp_df.iloc[0]["Feature"][:20])
                    with col2:
                        st.metric("Top Importance", f"{imp_df.iloc[0]['Importance']:.3f}")
                    with col3:
                        st.metric("Features Analyzed", len(importance))
    
    # LIME results
    if "lime" in explainability:
        lime_data = explainability["lime"]
        
        with st.expander("📈 LIME Analysis", expanded=True):
            if lime_data.get("success"):
                explanations = lime_data.get("explanations", {})
                
                if explanations:
                    exp_df = pd.DataFrame({
                        "Feature": list(explanations.keys()),
                        "Weight": list(explanations.values())
                    }).sort_values("Weight", key=abs, ascending=False).head(10)
                    
                    fig = px.bar(exp_df, x="Weight", y="Feature", orientation='h',
                               color="Weight", color_continuous_scale="RdBu",
                               title="Top 10 Feature Weights (LIME)")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    # Mixed or other methods
    if "mixed" in explainability:
        st.info("Mixed methods analysis combining SHAP and LIME")
    
    if "fast" in explainability:
        st.info("Fast heuristic analysis based on domain rules")

def display_validation_tab(validation_scores: Dict, embedding_scores: Dict, confidence: float):
    """Display full validation details"""
    st.markdown("### ✅ Detailed Validation Metrics")
    
    # All validation scores as chart
    if validation_scores:
        val_data = []
        for key, value in validation_scores.items():
            if isinstance(value, (int, float)) and key != "overall_confidence":
                val_data.append({
                    "Metric": key.replace("_", " ").title(),
                    "Score": value
                })
        
        if val_data:
            df = pd.DataFrame(val_data)
            fig = px.bar(df, x="Metric", y="Score", color="Score",
                        color_continuous_scale="Viridis",
                        title="All Validation Metrics")
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence gauge
    st.markdown("#### 🎯 Confidence Level")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Confidence (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Embedding scores
    if embedding_scores:
        st.markdown("#### 🔍 Embedding Similarity Scores")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Draft Similarity", f"{embedding_scores.get('cosine_draft_similarity', 0.0):.3f}")
        with col2:
            st.metric("Query Relevance", f"{embedding_scores.get('cosine_query_relevance', 0.0):.3f}")
        with col3:
            st.metric("Coherence", f"{embedding_scores.get('response_coherence', 0.0):.3f}")

def display_optimization_status():
    """Display optimization status with polling"""
    status = st.session_state.optimization_status
    
    if status == "running":
        st.info("🔄 Bayesian optimization running in background...")
        
        # Poll for status
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Check Status", key="check_opt_status"):
                try:
                    api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}"
                    opt_url = f"{api_url}/optimization/{st.session_state.session_id}"
                    response = requests.get(opt_url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "completed":
                            st.session_state.optimization_status = "completed"
                            st.session_state.analytics_result["optimization"] = data.get("result", {})
                            st.success("✅ Optimization complete!")
                            st.rerun()
                        else:
                            st.info(f"Optimization status: {data.get('status', 'unknown')}")
                    else:
                        st.warning("Could not fetch optimization status")
                except Exception as e:
                    st.warning(f"Connection error: {e}")
    
    elif status == "completed":
        st.success("✅ Optimization completed!")
        # Show optimization results if available
        optimization = st.session_state.analytics_result.get("optimization", {})
        if optimization:
            with st.expander("📈 Optimization Results", expanded=True):
                st.json(optimization)

def display_arxiv_tab(papers: List[Dict]):
    """Display arXiv papers"""
    if not papers:
        st.info("No papers available. Click 'Get Relevant Papers' button.")
        return
    
    st.markdown(f"### 📚 Relevant Papers ({len(papers)})")
    
    for i, paper in enumerate(papers):
        with st.expander(f"📄 {paper.get('title', 'Unknown')}", expanded=i == 0):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Authors:** {paper.get('authors', 'N/A')}")
                st.markdown(f"**Published:** {paper.get('published', 'N/A')}")
                st.markdown(f"**Summary:** {paper.get('summary', 'N/A')}")
                
                # Keywords or categories
                if 'categories' in paper:
                    st.markdown(f"**Categories:** {paper['categories']}")
            
            with col2:
                if 'pdf_url' in paper:
                    st.markdown(f"[📥 Download PDF]({paper['pdf_url']})")
                
                if 'arxiv_url' in paper:
                    st.markdown(f"[🔗 arXiv Page]({paper['arxiv_url']})")

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Domain selection
        domain_options = {
            "biomed": "🧬 Biomedical Science",
            "cs": "💻 Computer Science",
            "general": "🌐 General Research"
        }
        
        current_domain = st.session_state.domain
        selected_domain = st.selectbox(
            "Research Domain",
            options=list(domain_options.keys()),
            index=list(domain_options.keys()).index(current_domain) if current_domain in domain_options else 0,
            format_func=lambda x: domain_options[x]
        )
        
        if selected_domain != st.session_state.domain:
            st.session_state.domain = selected_domain
            # Test connection when domain changes
            st.session_state.api_connected = test_api_connection(selected_domain)
        
        st.markdown("---")
        
        # Connection status
        if not st.session_state.api_connected:
            st.error("❌ API Server Not Connected")
            st.info(f"Ensure server is running at: {BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}")
            if st.button("🔄 Test Connection", key="test_conn"):
                if test_api_connection(st.session_state.domain):
                    st.session_state.api_connected = True
                    st.success("✅ Connected!")
                    st.rerun()
                else:
                    st.error("❌ Still not connected")
        else:
            st.success("✅ API Connected")
        
        st.markdown("---")
        
        # Session info
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        st.markdown(f"**Parameters:** {len(st.session_state.extracted_parameters)}")
        
        if st.button("🔄 New Session", key="new_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        **IXORA** is a multi-agent research assistant that:
        - Extracts parameters from queries
        - Runs SHAP/LIME analytics
        - Performs Bayesian optimization
        - Provides causal analysis
        - Validates responses comprehensively
        """)
        
        st.markdown("---")
        st.markdown("#### 🔧 Quick Actions")
        
        if st.button("📋 View All Parameters", key="view_params"):
            st.session_state.active_tab = "parameters"
            st.rerun()
        
        if st.button("📊 View Analytics", key="view_analytics"):
            st.session_state.show_analytics = True
            st.rerun()

# ==================== MAIN CONTENT ====================
def render_main_content():
    st.markdown('<h1 class="main-title">🧪 IXORA Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-Agent AI for Scientific Research Questions</p>', unsafe_allow_html=True)
    
    # Show warning if not connected
    if not st.session_state.api_connected:
        st.warning("""
        ⚠️ **API Server Not Connected**
        
        Please ensure the IXORA API server is running. The application will have limited functionality.
        """)
    
    # Show validation summary at top if available
    if st.session_state.validation_scores or st.session_state.messages:
        display_validation_summary()
        st.markdown("---")
    
    # Show analytics method if available
    if st.session_state.analytics_method and st.session_state.analytics_method != "none":
        display_analytics_method_badge()
        st.markdown("---")
    
    # Show trace summary if available
    if st.session_state.trace_data:
        with st.expander("🧠 View AI Reasoning Trace", expanded=False):
            display_trace_summary()
        st.markdown("---")
    
    # Show optimization status
    if st.session_state.optimization_status != "not_started":
        display_optimization_status()
        st.markdown("---")
    
    # Chat history
    st.markdown("### 💬 Conversation")
    
    # Display messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {msg["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            content = msg["content"]
            sections = extract_xml_sections(content)
            
            if sections:
                display_text = ""
                for key in ["enthusiasm", "clarify", "explanation", "hypothesis", "followup"]:
                    if key in sections:
                        header = key.title()
                        if key == "enthusiasm":
                            header = "🌟 Enthusiasm"
                        elif key == "clarify":
                            header = "❓ Clarify"
                        elif key == "explanation":
                            header = "📚 Explanation"
                        elif key == "hypothesis":
                            header = "🔬 Hypothesis"
                        elif key == "followup":
                            header = "💡 Follow-up"
                        
                        display_text += f"**{header}:** {sections[key]}\n\n"
                
                st.markdown(f'<div class="assistant-message">{display_text}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{content}</div>', 
                           unsafe_allow_html=True)
            
            if "confidence" in msg:
                conf = msg["confidence"]
                conf_class = get_confidence_class(conf)
                st.markdown(f'<span class="confidence-badge confidence-{conf_class}">Confidence: {conf:.0%}</span>', 
                          unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask a research question...", disabled=not st.session_state.api_connected)
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="user-message"><strong>You:</strong> {user_input}</div>', 
                   unsafe_allow_html=True)
        
        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}/chat"
        
        with st.spinner("🔬 Analyzing your query..."):
            try:
                payload = {
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "domain": st.session_state.domain
                }
                response = requests.post(api_url, json=payload, timeout=210)
                response.raise_for_status()
                data = response.json()
                
                # Extract data
                response_text = data.get("response", "No response.")
                trace = data.get("trace", [])
                validation_scores = data.get("validation_scores", {})
                confidence = data.get("confidence", 0.7)
                parameters = data.get("parameters", {})
                
                # Get analytics info
                analytics_summary = data.get("analytics_summary", {})
                
                # Update session state
                st.session_state.last_response = data
                st.session_state.trace_data = trace
                st.session_state.validation_scores = validation_scores
                st.session_state.extracted_parameters = parameters
                st.session_state.embedding_scores = data.get("embedding_scores", {})
                st.session_state.show_analytics = True
                
                # Extract analytics method
                if "white_box_state" in data and "analytics" in data["white_box_state"]:
                    analytics = data["white_box_state"]["analytics"]
                    st.session_state.analytics_method = analytics.get("explainability_method", "none")
                    st.session_state.analytics_result = parse_analytics_result(analytics)
                else:
                    # Try to get analytics from other fields
                    if analytics_summary:
                        st.session_state.analytics_method = analytics_summary.get("method", "none")
                
                # Check optimization status
                if analytics_summary.get("bayesian_optimization") == "running":
                    st.session_state.optimization_status = "running"
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "confidence": confidence
                })
                
                st.rerun()
                
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try a simpler query.")
            except ConnectionError:
                st.error("🔌 Connection error. Check if API server is running.")
                st.session_state.api_connected = False
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error: {e}", exc_info=True)
    
    # On-demand buttons
    if st.session_state.extracted_parameters and st.session_state.api_connected:
        st.markdown("---")
        st.markdown("### 🔬 On-Demand Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⚡ Run Causal Analysis", use_container_width=True, type="primary"):
                with st.spinner("Running causal analysis..."):
                    try:
                        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}/causal"
                        payload = {
                            "query": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                            "session_id": st.session_state.session_id,
                            "domain": st.session_state.domain
                        }
                        response = requests.post(api_url, json=payload, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.causal_analysis = result.get("causal_results", {})
                            st.success("✅ Causal analysis complete!")
                            st.rerun()
                        else:
                            st.error("Failed to run causal analysis")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("📚 Get Relevant Papers", use_container_width=True, type="secondary"):
                with st.spinner("Searching arXiv..."):
                    try:
                        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}/arxiv"
                        query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
                        payload = {"query": query}
                        response = requests.post(api_url, json=payload, timeout=40)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.arxiv_papers = data.get("links", [])
                            st.success(f"✅ Found {len(st.session_state.arxiv_papers)} papers!")
                            st.rerun()
                        else:
                            st.error("ArXiv search failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col3:
            if st.button("⚙️ Run Optimization", use_container_width=True, type="secondary"):
                with st.spinner("Starting Bayesian optimization..."):
                    try:
                        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)}/optimize"
                        payload = {
                            "session_id": st.session_state.session_id,
                            "parameters": st.session_state.extracted_parameters
                        }
                        response = requests.post(api_url, json=payload, timeout=30)
                        
                        if response.status_code == 200:
                            st.session_state.optimization_status = "running"
                            st.success("✅ Optimization started!")
                            st.rerun()
                        else:
                            st.error("Failed to start optimization")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Analytics dashboard
    if st.session_state.show_analytics:
        st.markdown("---")
        st.markdown("### 📊 Detailed Analytics")
        
        tab_titles = ["📊 Explainability", "📈 Validation Details", "🔢 Parameters", "📚 Papers"]
        if st.session_state.causal_analysis:
            tab_titles.append("🔗 Causal Analysis")
        
        tabs = st.tabs(tab_titles)
        
        with tabs[0]:
            display_explainability_tab(st.session_state.analytics_result)
        
        with tabs[1]:
            display_validation_tab(
                st.session_state.validation_scores,
                st.session_state.embedding_scores,
                st.session_state.messages[-1].get("confidence", 0.7) if st.session_state.messages else 0.7
            )
        
        with tabs[2]:
            display_parameters_tab(st.session_state.extracted_parameters)
        
        with tabs[3]:
            display_arxiv_tab(st.session_state.arxiv_papers)
        
        # Causal analysis tab
        if st.session_state.causal_analysis and len(tabs) > 4:
            with tabs[4]:
                st.markdown("### 🔗 Causal Analysis Results")
                st.json(st.session_state.causal_analysis)

# ==================== MAIN ====================
def main():
    """Main application entry point"""
    init_session_state()
    
    # Test API connection on startup
    if "api_connected" not in st.session_state or st.session_state.api_connected is True:
        st.session_state.api_connected = test_api_connection(st.session_state.domain)
    
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()