"""
app.py - IXORA Biomedical AI Assistant
Complete Streamlit frontend with all features working properly
"""

import streamlit as st
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
import uuid
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
import logging
import pandas as pd
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost").rstrip("/")
DEFAULT_PORT = 8000
DOMAIN_PORTS = {
    "biomed": int(os.getenv("BIOMED_PORT", DEFAULT_PORT)),
    "general": int(os.getenv("GENERAL_PORT", DEFAULT_PORT))
}

# Set page configuration
st.set_page_config(
    page_title="🧪 IXORA - Biomedical AI Assistant",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main title styling */
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
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Message styling */
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
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        overflow-y: auto;
        max-height: 700px;
    }
    
    /* Parameter box styling */
    .param-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Confidence badge styling */
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-left: 0.5rem;
    }
    
    .confidence-high-badge {
        background: #4caf50;
        color: white;
    }
    
    .confidence-medium-badge {
        background: #ff9800;
        color: white;
    }
    
    .confidence-low-badge {
        background: #f44336;
        color: white;
    }
    
    /* Response section styling */
    .biomed-response {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .response-section {
        margin: 1.5rem 0;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .response-section h4 {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Special box styling */
    .hypothesis-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .followup-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Trace step styling */
    .trace-step {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .trace-error {
        border-left-color: #f44336;
        background: #ffebee;
    }
    
    .trace-success {
        border-left-color: #4caf50;
        background: #e8f5e9;
    }
    
    .trace-analytics {
        border-left-color: #2196f3;
        background: #e3f2fd;
    }
    
    .trace-hypothesis {
        border-left-color: #9c27b0;
        background: #f3e5f5;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = "biomed"
    
    if "last_response" not in st.session_state:
        st.session_state.last_response = {}
    
    if "links_results" not in st.session_state:
        st.session_state.links_results = None
    
    if "causal_results" not in st.session_state:
        st.session_state.causal_results = None
    
    if "validation_scores" not in st.session_state:
        st.session_state.validation_scores = {}
    
    if "trace_data" not in st.session_state:
        st.session_state.trace_data = []
    
    # ===== NEW: Persistent flags for tools visibility =====
    if "tools_visible" not in st.session_state:
        st.session_state.tools_visible = False
    
    if "show_analysis_tabs" not in st.session_state:
        st.session_state.show_analysis_tabs = False
    
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    
    if "api_url" not in st.session_state:
        st.session_state.api_url = ""
    
    if "extracted_parameters" not in st.session_state:
        st.session_state.extracted_parameters = {}


# ==================== HELPER FUNCTIONS ====================
def check_server_health(api_url):
    """Check if the server is online"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        return False, None
    except Exception:
        return False, None

def extract_response_sections(response_text):
    """Extract structured sections from XML-like response"""
    sections = {
        "enthusiasm": "",
        "explanation": "",
        "hypothesis": "",
        "followup": ""
    }
    
    for section in sections.keys():
        start_tag = f"<{section}>"
        end_tag = f"</{section}>"
        if start_tag in response_text and end_tag in response_text:
            start_idx = response_text.find(start_tag) + len(start_tag)
            end_idx = response_text.find(end_tag)
            sections[section] = response_text[start_idx:end_idx].strip()
    
    return sections

def display_assistant_response(response_text, confidence):
    """Display assistant response with proper formatting"""
    sections = extract_response_sections(response_text)
    
    # Determine confidence badge class
    if confidence > 0.8:
        conf_class = "confidence-high-badge"
    elif confidence > 0.6:
        conf_class = "confidence-medium-badge"
    else:
        conf_class = "confidence-low-badge"
    
    # Display response container
    st.markdown('<div class="biomed-response">', unsafe_allow_html=True)
    
    # Confidence badge
    st.markdown(f'<span class="confidence-badge {conf_class}">Confidence: {confidence:.1%}</span>', 
                unsafe_allow_html=True)
    
    st.markdown('<strong>🤖 IXORA:</strong><br>', unsafe_allow_html=True)
    
    # Display enthusiasm
    if sections["enthusiasm"]:
        st.markdown(f'### 🎉 {sections["enthusiasm"]}')
    
    # Display explanation
    if sections["explanation"]:
        st.markdown('<div class="response-section">', unsafe_allow_html=True)
        st.markdown("### 📚 Detailed Explanation")
        paragraphs = [p.strip() for p in sections["explanation"].split('\n\n') if p.strip()]
        for para in paragraphs:
            st.markdown(para)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display hypothesis
    if sections["hypothesis"]:
        st.markdown('<div class="hypothesis-box">', unsafe_allow_html=True)
        st.markdown("### 🧪 Testable Hypothesis")
        st.markdown(f'**{sections["hypothesis"]}**')
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display follow-up questions
    if sections["followup"]:
        st.markdown('<div class="followup-box">', unsafe_allow_html=True)
        st.markdown("### 🤔 Follow-up Questions")
        questions = [q.strip() for q in sections["followup"].split('\n') 
                    if q.strip() and q.strip().startswith(('1', '2', '3', '-', '•'))]
        for q in questions:
            st.markdown(f"- {q.lstrip('123.-• ')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # If no structure found, show raw text
    if not any(sections.values()):
        st.markdown(response_text)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_parameters_tab(parameters):
    """Display the parameters tab content"""
    if parameters:
        st.markdown(f"**Found {len(parameters)} parameters:**")
        
        # Display each parameter in a box
        for k, v in parameters.items():
            if isinstance(v, dict):
                val = v.get("value", "")
                if isinstance(val, list):
                    val = f"{val[0]} – {val[1]}" if len(val) == 2 else str(val)
                unit = v.get("unit", "")
                conf = v.get("confidence", 0.5)
                method = v.get("method", "unknown")
                
                # Determine color based on confidence
                if conf > 0.7:
                    color = "#4caf50"
                elif conf > 0.4:
                    color = "#ff9800"
                else:
                    color = "#f44336"
                
                st.markdown(f"""
                <div class="param-box">
                    <strong>🔬 {k.replace('_', ' ').title()}:</strong> 
                    <span style="color:#2196f3;font-size:1.2em">{val}</span> {unit}<br>
                    <small>Method: {method} • Confidence: <span style="color:{color}">{conf:.0%}</span></small>
                </div>
                """, unsafe_allow_html=True)
        
        # Create parameter summary table
        param_df = pd.DataFrame([
            {
                "Parameter": k,
                "Value": str(v.get("value", "")),
                "Unit": v.get("unit", ""),
                "Confidence": v.get("confidence", 0.5),
                "Method": v.get("method", "unknown")
            }
            for k, v in parameters.items()
        ])
        
        st.dataframe(param_df, use_container_width=True, hide_index=True)
    else:
        st.info("No parameters extracted from this query.")

def display_trace_tab(trace):
    """Display the AI trace tab content"""
    if trace:
        st.markdown("### 🔍 AI Reasoning Trace")
        
        # Summary metrics
        total_steps = len(trace)
        successful_steps = sum(1 for s in trace if "error" not in str(s.get("step", "")).lower())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Steps", total_steps)
        with col2:
            st.metric("Successful", successful_steps)
        with col3:
            total_time = sum(d.get("duration_ms", 0) for d in trace if isinstance(d.get("details"), dict))
            avg_time = total_time // max(1, total_steps)
            st.metric("Avg Time", f"{avg_time}ms")
        
        # Display each trace step
        for i, step in enumerate(trace):
            step_name = step.get("step", "Unknown")
            step_summary = step.get("summary", "")
            step_details = step.get("details", {})
            step_timestamp = step.get("timestamp", "")
            
            # Determine styling based on step type
            if "error" in step_name.lower():
                icon = "❌"
                trace_class = "trace-error"
            elif "validator" in step_name.lower():
                icon = "✅"
                trace_class = "trace-success"
            elif "analytics" in step_name.lower():
                icon = "📊"
                trace_class = "trace-analytics"
            elif "hypothesis" in step_name.lower():
                icon = "🧪"
                trace_class = "trace-hypothesis"
            else:
                icon = "⚡"
                trace_class = "trace-step"
            
            with st.expander(f"{icon} Step {i+1}: {step_name}", expanded=(i < 2)):
                st.markdown(f'<div class="{trace_class}">', unsafe_allow_html=True)
                
                if step_summary:
                    st.markdown(f"**Summary:** {step_summary}")
                
                if step_timestamp:
                    st.caption(f"⏱️ {step_timestamp}")
                
                # Display metrics if available
                if isinstance(step_details, dict) and step_details:
                    metrics_to_show = ["confidence", "parameters_count", "explainability_method", 
                                      "duration_ms", "samples", "improvement_pct", "ate", "p_value"]
                    
                    metrics_found = {}
                    for metric in metrics_to_show:
                        if metric in step_details:
                            metrics_found[metric] = step_details[metric]
                    
                    if metrics_found:
                        cols = st.columns(min(4, len(metrics_found)))
                        metric_items = list(metrics_found.items())
                        for idx in range(len(metric_items)):
                            metric_name, metric_value = metric_items[idx]
                            with cols[idx % len(cols)]:
                                if isinstance(metric_value, float):
                                    display_value = f"{metric_value:.3f}"
                                else:
                                    display_value = str(metric_value)
                                st.metric(
                                    metric_name.replace("_", " ").title(),
                                    display_value
                                )
                
                # Show full details
                if step_details:
                    with st.expander("View Details"):
                        st.json(step_details, expanded=False)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Trace timeline visualization
        st.markdown("---")
        st.markdown("#### 📈 Trace Timeline")
        
        timeline_data = []
        for i, step in enumerate(trace):
            step_name = step.get("step", f"Step {i+1}")
            duration = step.get("duration_ms", 0) / 1000  # Convert to seconds
            if duration == 0:
                duration = 0.5  # Default for steps without duration
            timeline_data.append({
                "Step": step_name,
                "Duration (s)": duration,
                "Status": "Error" if "error" in str(step).lower() else "Success"
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            st.bar_chart(timeline_df.set_index("Step")["Duration (s)"], use_container_width=True)
    else:
        st.info("No trace available for this response.")

def display_validation_tab(validation_scores, confidence):
    """Display the validation tab content"""
    if validation_scores:
        overall = validation_scores.get("overall_confidence", confidence)
        
        # Create two columns for better layout
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Display confidence with color and gauge
            conf_color = "#4caf50" if overall > 0.8 else "#ff9800" if overall > 0.6 else "#f44336"
            st.markdown(f"### 🎯 Overall Confidence: **<span style='color:{conf_color}'>{overall:.1%}</span>**", 
                       unsafe_allow_html=True)
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': conf_color},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            # Quick stats
            st.metric("Validation Score", f"{overall:.1%}")
            if validation_scores.get("cosine_query_relevance"):
                rel_score = validation_scores.get("cosine_query_relevance", 0)
                st.metric("Query Relevance", f"{rel_score:.1%}")
            if validation_scores.get("bertscore_f1"):
                qual_score = validation_scores.get("bertscore_f1", 0)
                st.metric("Content Quality", f"{qual_score:.1%}")
        
        # Detailed sections
        st.markdown("---")
        
        # Structural Validation
        with st.expander("🏗️ Structural Validation", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Enthusiasm", "✅" if validation_scores.get("has_enthusiasm") else "❌")
            with cols[1]:
                st.metric("Explanation", "✅" if validation_scores.get("has_explanation") else "❌")
            with cols[2]:
                st.metric("Hypothesis", "✅" if validation_scores.get("has_hypothesis") else "❌")
            with cols[3]:
                st.metric("Follow-up", "✅" if validation_scores.get("has_followup") else "❌")
            
            struct_score = validation_scores.get('structural_completeness', 0)
            st.progress(struct_score, text=f"Structural Completeness: {struct_score:.1%}")
        
        # Content Validation
        with st.expander("📊 Content Quality Metrics"):
            score_items = []
            for key, value in validation_scores.items():
                if isinstance(value, (int, float)) and key not in ['has_enthusiasm', 'has_explanation', 
                                                                  'has_hypothesis', 'has_followup']:
                    score_items.append({
                        "Metric": key.replace("_", " ").title(),
                        "Score": f"{value:.3f}",
                        "Percentage": f"{value*100:.1f}%" if value <= 1 else f"{value:.1f}",
                        "Status": "✅ Good" if (value > 0.7 if value <= 1 else value > 0.5) else "⚠️ Needs Improvement"
                    })
            
            if score_items:
                df = pd.DataFrame(score_items)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Add color-coded bars for visual assessment
            for item in score_items[:6]:
                score = float(item["Score"])
                color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{item['Metric']}</span>
                        <span>{item['Percentage']}</span>
                    </div>
                    <div style="background: #eee; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: {color}; width: {score*100}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # RLHF Status
        with st.expander("🤖 RLHF Status & Insights"):
            if validation_scores.get("rlhf_reward"):
                st.metric("RLHF Reward Score", f"{validation_scores.get('rlhf_reward', 0):.3f}")
            
            if validation_scores.get("rlhf_selected") == "alternative":
                st.success("✅ RLHF selected an improved response variant")
            else:
                st.info("ℹ️ Using primary response (RLHF active)")
            
            # Check RLHF training status
            try:
                api_url = f"{BASE_API_URL}:{DOMAIN_PORTS[st.session_state.selected_domain]}"
                rlhf_status = requests.get(f"{api_url}/rlhf/status", timeout=5).json()
                st.metric("Feedback Collected", rlhf_status.get("feedback_collected", 0))
                st.metric("Model Trained", "✅" if rlhf_status.get("model_trained") else "❌")
            except:
                st.info("RLHF status check unavailable")
        
        # Recommendations
        with st.expander("💡 Recommendations for Improvement"):
            recommendations = []
            
            if validation_scores.get("cosine_query_relevance", 1) < 0.6:
                recommendations.append("Improve relevance to the original query")
            
            if validation_scores.get("structural_completeness", 1) < 0.8:
                recommendations.append("Ensure all response sections are present")
            
            if validation_scores.get("response_length", 0) < 0.3:
                recommendations.append("Provide more detailed explanations")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            else:
                st.success("✅ Response meets all quality standards!")
    else:
        st.info("No validation scores available for this response.")
        st.markdown("Validation scores are generated during the response generation process.")

def get_api_url():
    """
    Return the correct API URL based on the current domain selected in sidebar.
    Falls back to the last known working URL or default.
    """
    # Priority 1: Use the currently selected domain from session state
    if "selected_domain" in st.session_state:
        domain = st.session_state.selected_domain
        port = DOMAIN_PORTS.get(domain, DEFAULT_PORT)
        return f"http://localhost:{port}"
    
    # Priority 2: Use the last known working URL
    if "last_working_url" in st.session_state:
        return st.session_state.last_working_url
    
    # Priority 3: Default to biomed
    return f"http://localhost:{DOMAIN_PORTS['biomed']}"


# ===== COMPLETELY REWRITTEN display_tools_tab function =====
def display_tools_tab():
    """🧰 Tools tab - FIXED with proper Streamlit state handling"""
    st.markdown("### 🧰 On-Demand Research Tools")
    
    # Get from session state (always available)
    api_url = st.session_state.api_url
    session_id = st.session_state.session_id
    
    # Debug info
    st.caption(f"🔧 Session: {session_id[:8]}... | Backend: {api_url}")
    
    # ===== ADD DEBUG OUTPUT =====
    # st.write("🔍 DEBUG: display_tools_tab() is being rendered")  # Uncomment to debug
    
    col1, col2 = st.columns(2)
    
    # ====================== ARXIV SEARCH ======================
    with col1:
        st.markdown("#### 📚 ArXiv Evidence Search")
        st.caption("Search arXiv for papers related to your research query.")
        
        # Button with unique key
        arxiv_clicked = st.button(
            "🔍 Search arXiv Papers", 
            key="arxiv_search_button",
            use_container_width=True
        )
        
        # ===== DEBUG OUTPUT =====
        # st.write(f"DEBUG: arxiv_clicked = {arxiv_clicked}")  # Uncomment to debug
        
        # Handle click immediately in same run
        if arxiv_clicked:
            st.write("🔄 Button clicked! Starting arXiv search...")
            
            with st.spinner("Searching arXiv... (up to 45 seconds)"):
                try:
                    payload = {"session_id": session_id}
                    
                    st.write(f"📤 Sending POST to: `{api_url}/arxiv`")
                    
                    response = requests.post(
                        f"{api_url}/arxiv",
                        json=payload,
                        timeout=120
                    )
                    
                    st.write(f"📥 Response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("status") == "success":
                            papers = data.get("links", [])
                            st.success(f"✅ Found {len(papers)} papers!")
                            
                            # Store in session state for persistence
                            st.session_state.links_results = papers
                            
                            for i, paper in enumerate(papers, 1):
                                with st.expander(f"{i}. {paper['title'][:80]}..."):
                                    st.write(f"**Authors**: {paper['authors']}")
                                    st.write(f"**Published**: {paper['published']}")
                                    st.write(f"**Summary**: {paper['summary'][:300]}...")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        if paper.get("url"):
                                            st.markdown(f"[📄 View]({paper['url']})")
                                    with col_b:
                                        if paper.get("pdf_url"):
                                            st.markdown(f"[📥 PDF]({paper['pdf_url']})")
                        else:
                            st.error(f"❌ {data.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Server error {response.status_code}")
                        st.code(response.text[:300])
                
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout after 50 seconds")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Cannot connect to {api_url}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Show cached results if available
        elif st.session_state.links_results:
            st.info("📚 Previous arXiv results (click button to refresh)")
            with st.expander("View cached results"):
                for i, paper in enumerate(st.session_state.links_results[:3], 1):
                    st.write(f"{i}. {paper['title'][:60]}...")
    
    # ====================== CAUSAL ANALYSIS ======================
    with col2:
        st.markdown("#### 🔬 Causal Inference")
        st.caption("Run causal analysis on extracted parameters.")
        
        # Button with unique key
        causal_clicked = st.button(
            "🧠 Run Causal Analysis",
            key="causal_analysis_button",
            use_container_width=True
        )
        
        # ===== DEBUG OUTPUT =====
        # st.write(f"DEBUG: causal_clicked = {causal_clicked}")  # Uncomment to debug
        
        # Handle click immediately
        if causal_clicked:
            st.write("🔄 Button clicked! Starting causal analysis...")
            
            with st.spinner("Running causal analysis... (up to 60 seconds)"):
                try:
                    payload = {"session_id": session_id}
                    
                    st.write(f"📤 Sending POST to: `{api_url}/causal`")
                    
                    response = requests.post(
                        f"{api_url}/causal",
                        json=payload,
                        timeout=70
                    )
                    
                    st.write(f"📥 Response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("status") == "success":
                            causal = data.get("causal", {})
                            st.success("✅ Causal analysis complete!")
                            
                            # Store in session state
                            st.session_state.causal_results = causal
                            
                            # Display results
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if "ate" in causal:
                                    st.metric("ATE", f"{causal['ate']:.3f}")
                            with col_b:
                                if "p_value" in causal:
                                    st.metric("p-value", f"{causal['p_value']:.4f}")
                            
                            if "ci_95_lower" in causal:
                                st.write(f"**95% CI**: [{causal['ci_95_lower']:.3f}, {causal['ci_95_upper']:.3f}]")
                        else:
                            st.error(f"❌ {data.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Server error {response.status_code}")
                        st.code(response.text[:300])
                
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout after 70 seconds")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Cannot connect to {api_url}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Show cached results if available
        elif st.session_state.causal_results:
            st.info("🔬 Previous causal results (click button to refresh)")
            with st.expander("View cached results"):
                causal = st.session_state.causal_results
                st.write(f"ATE: {causal.get('ate', 0):.3f}")
                st.write(f"p-value: {causal.get('p_value', 1):.4f}")
    
    # Debug section
    with st.expander("🔧 Debug Info"):
        st.write(f"**Session ID**: `{session_id}`")
        st.write(f"**API URL**: `{api_url}`")
        st.write(f"**Last Query**: `{st.session_state.last_query[:100]}...`")
        st.write(f"**Tools Visible**: {st.session_state.tools_visible}")
        st.write(f"**Tabs Visible**: {st.session_state.show_analysis_tabs}")
        
        # Test backend
        if st.button("🧪 Test Backend", key="test_backend_connection"):
            try:
                test_resp = requests.get(f"{api_url}/health", timeout=5)
                if test_resp.status_code == 200:
                    st.success("✅ Backend is online!")
                    st.json(test_resp.json())
                else:
                    st.error(f"❌ Backend error {test_resp.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

def send_feedback(preference):
    """Send feedback to the server"""
    try:
        api_url = f"{BASE_API_URL}:{DOMAIN_PORTS[st.session_state.selected_domain]}"
        response = requests.post(f"{api_url}/feedback", json={
            "session_id": st.session_state.session_id,
            "preference": preference,
            "response": st.session_state.last_response["response"],
            "query_hash": st.session_state.last_response.get("query_hash", "unknown")
        }, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# ==================== SIDEBAR ====================
def render_sidebar():
    """Render the sidebar content"""
    st.sidebar.title("🧪 IXORA Settings")
    
    # Domain selection
    domain = st.sidebar.selectbox("Domain", ["biomed", "general"], 
                                  index=0 if st.session_state.selected_domain == "biomed" else 1)
    if domain != st.session_state.selected_domain:
        st.session_state.selected_domain = domain
        st.sidebar.success(f"Switched to {domain.upper()} domain")
    
    # Server health check
    api_url = f"{BASE_API_URL}:{DOMAIN_PORTS[domain]}"
    server_online, health_data = check_server_health(api_url)
    
    health_status = st.sidebar.container()
    if server_online:
        health_status.success(f"✅ {domain.upper()} Server Online")
        health_status.caption(f"Version: {health_data.get('version', 'unknown')}")
    else:
        health_status.error(f"❌ {domain.upper()} Server Offline")
    
    # Feedback section
    st.sidebar.markdown("### 📊 Help Improve Responses (RLHF)")
    if st.session_state.last_response.get("response"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("👍 Good", key="good", use_container_width=True):
                if send_feedback("good"):
                    st.sidebar.success("Thank you! 👍")
                else:
                    st.sidebar.error("Failed to send feedback")
        with col2:
            if st.button("👎 Bad", key="bad", use_container_width=True):
                if send_feedback("bad"):
                    st.sidebar.success("Thank you! 👎")
                else:
                    st.sidebar.error("Failed to send feedback")
    else:
        st.sidebar.info("Ask a question first to provide feedback")
    
    # Clear conversation
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        keys = ["messages", "last_response", "links_results", "causal_results", 
                "validation_scores", "trace_data", "arxiv_results_visible", 
                "causal_results_visible"]
        
        for k in keys:
            if "messages" in k:
                st.session_state[k] = []
            elif "trace_data" in k:
                st.session_state[k] = None
            else:
                st.session_state[k] = {}
        
        st.rerun()
    
    # Debug information
    with st.sidebar.expander("🔧 Debug Information"):
        st.write(f"Session ID: {st.session_state.session_id}")
        st.write(f"API URL: {api_url}")
        st.write(f"Messages: {len(st.session_state.messages)}")
        st.write(f"Server: {'Online' if server_online else 'Offline'}")
        if st.session_state.last_response:
            st.write(f"Last confidence: {st.session_state.last_response.get('confidence', 0):.1%}")
            st.write(f"Last trace steps: {len(st.session_state.last_response.get('trace', []))}")
    
    return server_online, api_url

# ==================== MAIN UI ====================
def render_main_content(server_online, api_url):
    """Render the main content area - FIXED VERSION"""
    
    # ===== CRITICAL: Store API URL in session state =====
    st.session_state.api_url = api_url
    
    # Header
    st.markdown('<h1 class="main-title">🧪 IXORA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-Agent Biomedical Research Assistant</p>', unsafe_allow_html=True)
    
    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{msg["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            response_text = msg["content"]
            confidence = msg.get("confidence", 0.7)
            display_assistant_response(response_text, confidence)
    
    # Chat input
    prompt = st.chat_input("Ask a biomedical research question...", disabled=not server_online)
    
    if prompt:
        if not server_online:
            st.error("Server offline. Please check if the backend is running.")
        else:
            # ===== Store query for tools =====
            st.session_state.last_query = prompt
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{prompt}</div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("🧠 Analyzing... (up to 180s)"):
                try:
                    # Make API call
                    response = requests.post(
                        f"{api_url}/chat",
                        json={"message": prompt, "session_id": st.session_state.session_id},
                        timeout=190
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract data
                    response_text = data.get("response", "No response generated.")
                    confidence = data.get("confidence", 0.7)
                    trace = data.get("trace", [])
                    validation_scores = data.get("validation_scores", {})
                    parameters = data.get("extracted_parameters", {})
                    
                    # Store in session state
                    st.session_state.last_response = data
                    st.session_state.validation_scores = validation_scores
                    st.session_state.trace_data = trace
                    st.session_state.extracted_parameters = parameters
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text, 
                        "confidence": confidence
                    })
                    
                    # ===== CRITICAL: Enable tabs and tools =====
                    st.session_state.show_analysis_tabs = True
                    st.session_state.tools_visible = True
                    
                    # Display response
                    display_assistant_response(response_text, confidence)
                    st.success(f"✅ Analysis complete! (Confidence: {confidence:.1%})")
                    
                except requests.exceptions.Timeout:
                    st.error("Request timeout. The server took too long to respond.")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error. Please check if the backend server is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)[:200]}")
    
    # ===== UNCONDITIONAL TAB RENDERING (THE FIX!) =====
    if st.session_state.show_analysis_tabs:
        st.markdown("### 📊 Analysis Results")
        
        # Get data from session state
        parameters = st.session_state.extracted_parameters
        trace = st.session_state.trace_data
        validation_scores = st.session_state.validation_scores
        confidence = st.session_state.last_response.get("confidence", 0.7)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧪 Parameters", 
            "🔍 AI Trace", 
            "📊 Validation", 
            "🔬 Tools"
        ])
        
        with tab1:
            display_parameters_tab(parameters)
        
        with tab2:
            display_trace_tab(trace)
        
        with tab3:
            display_validation_tab(validation_scores, confidence)
        
        with tab4:
            # ===== ALWAYS RENDER TOOLS TAB =====
            if st.session_state.tools_visible:
                display_tools_tab()
            else:
                st.info("💡 Ask a question first to enable research tools.")
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("""
    <strong>🧪 IXORA • Multi-Agent Biomedical Research Assistant</strong><br>
    <em>Parameter extraction • Reasoning trace • Validation • On-demand analytics • RLHF</em>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# ==================== MAIN APPLICATION ====================
def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    server_online, api_url = render_sidebar()
    
    # Render main content
    render_main_content(server_online, api_url)

# Run the application
if __name__ == "__main__":
    main()