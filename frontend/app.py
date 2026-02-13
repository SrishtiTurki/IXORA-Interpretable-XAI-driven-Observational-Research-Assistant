"""
app.py - IXORA Research Assistant - ENHANCED VERSION WITH XML PARSING
"""

import streamlit as st
import requests
from requests.exceptions import Timeout, ConnectionError
import uuid
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import time
import hashlib
import httpx
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

# ==================== XML PARSING FUNCTIONS ====================
def parse_xml_response(response_text: str) -> Dict[str, str]:
    """Robust XML parsing with fallbacks"""
    sections = {
        "raw": response_text,
        "enthusiasm": "",
        "clarify": "",
        "explanation": "",
        "hypothesis": "",
        "followup": ""
    }
    
    # If text is empty, return empty sections
    if not response_text or not isinstance(response_text, str):
        return sections
    
    # Check if we even have XML tags
    if "<" not in response_text or ">" not in response_text:
        # No XML found, treat everything as explanation
        sections["explanation"] = response_text
        return sections
    
    # Parse each tag more carefully
    for tag in ["enthusiasm", "clarify", "explanation", "hypothesis", "followup"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_idx = response_text.find(start_tag)
        end_idx = response_text.find(end_tag)
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # Extract content
            content_start = start_idx + len(start_tag)
            content = response_text[content_start:end_idx].strip()
            
            # Clean up any nested tags or artifacts
            # Remove any remaining XML-like tags
            content = re.sub(r'<[^>]+>', '', content)
            sections[tag] = content
    
    # FALLBACK: If no explanation found but we have raw text
    if not sections["explanation"] and response_text:
        # Try to extract meaningful content
        lines = response_text.split('\n')
        meaningful_lines = [line.strip() for line in lines if line.strip() and not line.startswith('<')]
        if meaningful_lines:
            sections["explanation"] = '\n\n'.join(meaningful_lines[:10])  # First 10 lines
    
    return sections

def render_parsed_response(sections: Dict[str, str], domain: str = "biomed"):
    """
    Render a parsed XML response with nice formatting.
    
    Args:
        sections: Dict from parse_xml_response
        domain: "biomed" or "cs"
    """
    
    # 1. Enthusiasm section (always first if present)
    if sections.get("enthusiasm"):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
            <strong>✨ {sections['enthusiasm']}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Clarify section (CS domain - show prominently)
    if sections.get("clarify") and domain == "cs":
        st.markdown("### 🤔 Clarifying Questions")
        st.info(sections["clarify"])
        st.markdown("")
    
    # 3. Main explanation (core content)
    if sections.get("explanation"):
        st.markdown("### 📚 Detailed Explanation")
        
        # Clean and format the explanation
        explanation = sections["explanation"]
        
        # Remove any remaining stray tags
        explanation = explanation.replace("<explanation>", "").replace("</explanation>", "")
        
        # Format for better readability
        paragraphs = [p.strip() for p in explanation.split("\n\n") if p.strip()]
        for paragraph in paragraphs:
            # Check if paragraph looks like a heading
            if paragraph.startswith("**") and paragraph.endswith("**"):
                st.markdown(f"#### {paragraph}")
            else:
                st.markdown(paragraph)
                st.markdown("")  # Add spacing between paragraphs
    else:
        # Fallback: show raw text if no explanation section found
        st.markdown("### 📚 Response")
        st.markdown(sections.get("raw", ""))
    
    # 4. Hypothesis section (Biomed domain)
    if sections.get("hypothesis") and domain == "biomed":
        st.markdown("### 🔬 Testable Hypothesis")
        st.info(sections["hypothesis"])
        st.markdown("")
    
    # 5. Follow-up questions
    if sections.get("followup"):
        with st.expander("❓ Follow-up Questions", expanded=False):
            st.markdown(sections["followup"])



def render_parsed_response(sections: Dict[str, str], domain: str = "biomed"):
    """
    Render a parsed XML response with nice formatting.
    
    Args:
        sections: Dict from parse_xml_response
        domain: "biomed" or "cs"
    """
    
    # 1. Enthusiasm section (always first if present)
    if sections.get("enthusiasm"):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
            <strong>✨ {sections['enthusiasm']}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Clarify section (CS domain - show prominently)
    if sections.get("clarify") and domain == "cs":
        with st.expander("🤔 Clarifying Questions", expanded=True):
            st.markdown(sections["clarify"])
        st.markdown("")
    
    # 3. Main explanation (core content)
    if sections.get("explanation"):
        st.markdown("### 📚 Detailed Explanation")
        st.markdown(sections["explanation"])
        st.markdown("")
    
    # 4. Hypothesis section (Biomed domain)
    if sections.get("hypothesis") and domain == "biomed":
        st.markdown("### 🔬 Testable Hypothesis")
        st.info(sections["hypothesis"])
        st.markdown("")
    
    # 5. Follow-up questions
    if sections.get("followup"):
        with st.expander("❓ Follow-up Questions", expanded=False):
            st.markdown(sections["followup"])
    
    # Fallback: If no XML tags found, show raw response
    if not any([sections.get("enthusiasm"), sections.get("explanation")]):
        st.markdown(sections["raw"])


# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #667eea;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .section-title {
        color: #444;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .subsection-title {
        color: #555;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.2rem 0 0.8rem 0;
    }
    .card-title {
        color: #333;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .param-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.9rem;
        border-radius: 6px;
        margin: 0.6rem 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }
    .confidence-high { background: #4caf50; color: white; }
    .confidence-medium { background: #ff9800; color: white; }
    .confidence-low { background: #f44336; color: white; }
    .cot-bubble {
        background: #f0f7ff;
        border-left: 3px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .lime-explanation {
        background: #fff3e0;
        border-left: 3px solid #ff9800;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    .validation-metric {
        display: inline-block;
        background: #e3f2fd;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .action-button {
        width: 100%;
        margin: 0.3rem 0;
    }
    .dashboard-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .nav-button {
        margin: 0.3rem 0;
        border-radius: 8px;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background: #f0f7ff;
        border-color: #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INIT ====================
def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "domain": "biomed",
        "extracted_parameters": {},
        "analytics_result": None,
        "causal_analysis": None,
        "arxiv_papers": [],
        "optimization_status": None,
        "show_analytics": False,
        "api_connected": False,
        "last_response": None,
        "detailed_trace": [],
        "validation_metrics": {},
        "explainability_results": {},
        "active_tab": "chat",
        "optimization_polling": False,
        "background_tasks": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== API HELPERS ====================
def api_post(endpoint: str, payload: dict, timeout: int = 200) -> Optional[dict]:
    port = DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)
    url = f"{BASE_API_URL}:{port}/{endpoint}"
    logger.info(f"API POST to {url}")
    
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"POST failed: {e}")
        return {"error": str(e)}

def api_get(endpoint: str, timeout: int = 30) -> Optional[dict]:
    port = DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)
    url = f"{BASE_API_URL}:{port}/{endpoint}"
    logger.info(f"API GET to {url}")
    
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"GET failed: {e}")
        return {"error": str(e)}

def test_backend_connection():
    port = DOMAIN_PORTS.get(st.session_state.domain, DEFAULT_PORT)
    url = f"{BASE_API_URL}:{port}/health"
    
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            st.session_state.api_connected = True
            return True, resp.json()
        else:
            st.session_state.api_connected = False
            return False, f"HTTP {resp.status_code}"
    except Exception as e:
        st.session_state.api_connected = False
        return False, str(e)

# ==================== DISPLAY HELPERS ====================
def display_parameters(params: Dict):
    if not params:
        st.info("No parameters extracted.")
        return
    
    st.markdown("#### 📋 Extracted Parameters")
    
    for name, param in params.items():
        if isinstance(param, dict):
            val = param.get("value", "")
            unit = param.get("unit", "")
            conf = param.get("confidence", 0.5)
            method = param.get("method", "extracted")
        else:
            val = param
            unit = ""
            conf = 0.8
            method = "auto"
        
        if isinstance(val, list) and len(val) == 2:
            val_display = f"{val[0]} – {val[1]}"
        else:
            val_display = str(val)
        
        conf_class = "high" if conf >= 0.85 else "medium" if conf >= 0.6 else "low"
        badge = f'<span class="confidence-badge confidence-{conf_class}">{conf:.0%}</span>'
        
        st.markdown(
            f'<div class="param-box">'
            f'<strong>{name}</strong>: {val_display} {unit}<br>'
            f'<small>Confidence: {badge} | Method: {method}</small></div>',
            unsafe_allow_html=True
        )

def safe_to_progress_value(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if isinstance(v, str):
            cleaned = v.strip().rstrip("%")
            num = float(cleaned)
        else:
            num = float(v)

        if 0 <= num <= 1:
            return num
        elif 0 <= num <= 100:
            return num / 100.0
        else:
            return None
    except (ValueError, TypeError):
        return None

def display_enhanced_trace(trace_list):
    if not trace_list or not isinstance(trace_list, list):
        st.info("No detailed trace available yet.")
        return
    
    st.markdown("#### 🔍 Pipeline Trace")
    
    total_time = sum(float(step.get("time_seconds", 0)) for step in trace_list)
    success_count = sum(1 for step in trace_list if step.get("success", True))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Time", f"{total_time:.1f}s")
    with col2:
        st.metric("Success Rate", f"{success_count}/{len(trace_list)}")

    for idx, step in enumerate(trace_list, 1):
        step_name = step.get("step", "unknown").replace("_", " ").title()
        success = step.get("success", True)
        time_taken = step.get("time_seconds", None)

        icon = "✅" if success else "❌"

        with st.expander(f"Step {idx} • {icon} {step_name}", expanded=(idx == len(trace_list))):
            cols = st.columns([4, 1.5, 1.5])

            with cols[0]:
                method = step.get("method") or step.get("model") or step.get("methods")
                if method:
                    if isinstance(method, list):
                        method = ", ".join(str(x) for x in method)
                    st.markdown(f"**Method**: {method}")

                if "param_count" in step:
                    st.markdown(f"• **Parameters**: {step['param_count']}")
                if "token_count" in step:
                    st.markdown(f"• **Tokens**: {step['token_count']}")

            with cols[1]:
                if time_taken is not None:
                    st.metric("Duration", f"{float(time_taken):.1f} s")

            with cols[2]:
                status_color = "green" if success else "red"
                st.markdown(f"**Status**: :{status_color}[{'Success' if success else 'Failed'}]")

            with st.expander("Raw step data", expanded=False):
                st.json(step)

            if step.get("step") == "validation" and "embedding_scores" in step:
                st.markdown("**Validation Signals**")
                scores = step["embedding_scores"]
                if isinstance(scores, dict):
                    for key, raw in scores.items():
                        nice_key = key.replace("_", " ").title()
                        val = safe_to_progress_value(raw)
                        if val is not None:
                            st.progress(val)
                            st.caption(f"{nice_key}: **{val:.1%}**")
                        else:
                            st.caption(f"{nice_key}: **{raw}** (not plottable)")

def display_cot_steps(cot_data):
    if not cot_data:
        return
    
    st.markdown("#### 🧠 Chain-of-Thought")
    
    with st.expander("View Reasoning Steps", expanded=False):
        for i, step in enumerate(cot_data, 1):
            text = step if isinstance(step, str) else step.get("reasoning", str(step))
            st.markdown(f'<div class="cot-bubble"><strong>Step {i}:</strong> {text}</div>', unsafe_allow_html=True)

def display_explainability_results(explainability: Dict):
    """Display SHAP and LIME explainability results"""
    if not explainability:
        st.info("No explainability analysis available.")
        return
    
    st.markdown("#### 📊 Explainability Analysis")
    
    # SHAP Results
    if "shap" in explainability:
        shap_data = explainability["shap"]
        with st.expander("SHAP Feature Importance", expanded=False):
            st.markdown("**SHAP (SHapley Additive exPlanations)** explains how each feature contributes to the model's prediction.")
            
            if "importance" in shap_data:
                importance = shap_data["importance"]
                if isinstance(importance, dict):
                    # Create bar chart
                    df = pd.DataFrame({
                        "Feature": list(importance.keys()),
                        "Importance": list(importance.values())
                    }).sort_values("Importance", ascending=True)
                    
                    # Display as bars
                    for _, row in df.iterrows():
                        width = min(100, abs(row["Importance"]) * 100)
                        color = "#4caf50" if row["Importance"] > 0 else "#f44336"
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <strong>{row['Feature']}</strong>
                            <div style="width: 100%; background: #e0e0e0; border-radius: 4px;">
                                <div style="width: {width}%; background: {color}; height: 24px; border-radius: 4px; 
                                         display: flex; align-items: center; padding-left: 10px; color: white;">
                                    {row['Importance']:.3f}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if "interpretation" in shap_data:
                st.markdown("**Interpretation:**")
                st.info(shap_data["interpretation"])
    
    # LIME Results
    if "lime" in explainability:
        lime_data = explainability["lime"]
        with st.expander("LIME Local Explanations", expanded=False):
            st.markdown("**LIME (Local Interpretable Model-agnostic Explanations)** explains individual predictions by approximating the model locally.")
            
            if "explanations" in lime_data:
                explanations = lime_data["explanations"]
                if isinstance(explanations, dict):
                    for feature, weight in explanations.items():
                        effect = "Positive" if weight > 0 else "Negative"
                        color = "#4caf50" if weight > 0 else "#f44336"
                        st.markdown(f"""
                        <div class="lime-explanation">
                            <strong>{feature}</strong>: {effect} impact ({weight:.3f})
                            <div style="margin-left: 20px; font-size: 0.9em; color: #666;">
                                Weight: <span style="color: {color}; font-weight: bold;">{weight:.3f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if "interpretation" in lime_data:
                st.markdown("**Interpretation:**")
                st.info(lime_data["interpretation"])

def display_validation_metrics(validation: Dict):
    """Display comprehensive validation metrics"""
    if not validation:
        st.info("No validation metrics available.")
        return
    
    st.markdown("#### 🔍 Validation Metrics")
    
    with st.expander("View Validation Details", expanded=False):
        # Confidence score
        if "confidence" in validation:
            conf = validation["confidence"]
            conf_class = "high" if conf >= 0.85 else "medium" if conf >= 0.6 else "low"
            st.markdown(f'<span class="confidence-badge confidence-{conf_class}">Overall Confidence: {conf:.1%}</span>', 
                       unsafe_allow_html=True)
        
        # Embedding scores
        if "embedding_scores" in validation:
            emb_scores = validation["embedding_scores"]
            st.markdown("**Embedding Similarity Scores**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "cosine_draft_similarity" in emb_scores:
                    st.metric("Draft Similarity", f"{emb_scores['cosine_draft_similarity']:.3f}")
            with col2:
                if "cosine_query_relevance" in emb_scores:
                    st.metric("Query Relevance", f"{emb_scores['cosine_query_relevance']:.3f}")
            with col3:
                if "response_coherence" in emb_scores:
                    st.metric("Coherence", f"{emb_scores['response_coherence']:.3f}")
            
            # RLHF metrics
            if "rlhf_reward" in emb_scores:
                st.markdown("**RLHF (Reinforcement Learning from Human Feedback)**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Main Response Reward", f"{emb_scores['rlhf_reward']:.3f}")
                if "rlhf_comparison" in emb_scores:
                    with col2:
                        st.metric("Alternative Reward", f"{emb_scores['rlhf_comparison']:.3f}")
        
        # Additional metrics
        if "validation_scores" in validation:
            val_scores = validation["validation_scores"]
            if isinstance(val_scores, dict):
                st.markdown("**Additional Metrics:**")
                metrics_html = ""
                for key, value in val_scores.items():
                    if isinstance(value, (int, float)):
                        metrics_html += f'<span class="validation-metric">{key}: {value:.3f}</span>'
                if metrics_html:
                    st.markdown(metrics_html, unsafe_allow_html=True)

def display_arxiv_tab(papers: List):
    """Display arXiv papers"""
    if not papers:
        st.info("No papers found.")
        return
    
    st.markdown("### 📚 Literature Search Results")
    
    for i, p in enumerate(papers[:5]):
        with st.container():
            st.markdown(f"**{p.get('title', 'Untitled')}**")
            st.caption(f"👥 {p.get('authors', '')} • 📅 {p.get('published', '')}")
            
            summary = p.get('summary', '')
            if len(summary) > 200:
                with st.expander("Abstract", expanded=False):
                    st.write(summary)
            else:
                st.write(summary)
            
            # Safe URL handling
            pdf_url = p.get('pdf_url', '')
            if pdf_url:
                st.markdown(f"[📄 PDF]({pdf_url})")
            
            if i < len(papers[:5]) - 1:
                st.markdown("---")

def display_optimization_results():
    """Display Bayesian optimization results"""
    if not st.session_state.get("optimization_status"):
        st.info("No optimization results available.")
        return
    
    opt_data = st.session_state.optimization_status
    
    st.markdown("### 🔬 Bayesian Optimization Results")
    
    with st.container():
        if "status" in opt_data:
            status = opt_data["status"]
            if status == "completed":
                st.success("✅ Optimization completed successfully!")
            elif status == "running":
                st.info("🔄 Optimization in progress...")
                st.progress(0.7)
            elif status == "failed":
                st.error("❌ Optimization failed")
            elif status == "timeout":
                st.warning("⏰ Optimization timed out")
        
        # Display optimal parameters
        if "optimal_parameters" in opt_data and opt_data["optimal_parameters"]:
            st.markdown("#### Optimal Parameters Found:")
            optimal_params = opt_data["optimal_parameters"]
            if isinstance(optimal_params, dict):
                for param, value in optimal_params.items():
                    st.markdown(f"**{param}**: `{value}`")
            else:
                st.write(optimal_params)
        
        # Display best score
        if "best_score" in opt_data:
            st.metric("Best Score", f"{opt_data['best_score']:.4f}")
        
        # Display additional info
        if "n_evaluations" in opt_data:
            st.caption(f"Evaluations: {opt_data['n_evaluations']}")
        if "message" in opt_data:
            st.info(opt_data["message"])
        
        # Raw data expander
        with st.expander("📊 Raw Optimization Data"):
            st.json(opt_data)

def display_causal_analysis():
    """Display causal analysis results"""
    if not st.session_state.get("causal_analysis"):
        st.info("No causal analysis results available.")
        return
    
    causal_data = st.session_state.causal_analysis
    
    st.markdown("### ⚖️ Causal Analysis Results")
    
    with st.container():
        if "status" in causal_data:
            status = causal_data["status"]
            if status == "success":
                st.success("✅ Causal analysis completed")
            elif status == "simulated":
                st.info("📊 Simulated causal results (DoWhy not available)")
            elif status == "failed":
                st.error("❌ Causal analysis failed")
        
        # Display treatment and outcome
        if "treatment" in causal_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Treatment Variable", causal_data["treatment"])
            with col2:
                if "outcome" in causal_data:
                    st.metric("Outcome Variable", causal_data["outcome"])
        
        # Display effect estimates
        if "estimated_effect" in causal_data:
            effect = causal_data["estimated_effect"]
            if isinstance(effect, dict):
                if "value" in effect:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Treatment Effect", f"{effect['value']:.3f}")
                    with col2:
                        if "ci_low" in effect and "ci_high" in effect:
                            st.metric("95% CI", f"[{effect['ci_low']:.3f}, {effect['ci_high']:.3f}]")
        
        # Display plain explanation
        if "plain_explanation" in causal_data:
            with st.expander("📝 Interpretation", expanded=True):
                st.markdown(causal_data["plain_explanation"])
        
        # Raw data expander
        with st.expander("📊 Raw Causal Data"):
            st.json(causal_data)

# ==================== ACTION BUTTONS ====================
def render_action_buttons():
    """Render action buttons for additional analyses"""
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Run Causal Analysis", use_container_width=True, key="causal_btn"):
            run_causal_analysis_action()
    
    with col2:
        if st.button("📈 Check Optimization", use_container_width=True, key="opt_btn"):
            check_optimization_status()
    
    with col3:
        if st.button("📚 Search Literature", use_container_width=True, key="arxiv_btn"):
            run_arxiv_search()

def run_causal_analysis_action():
    """Trigger causal analysis"""
    if not st.session_state.get("last_response") or "parameters" not in st.session_state.last_response:
        st.warning("Please run a query first to extract parameters.")
        return
    
    with st.spinner("Running causal analysis..."):
        params = st.session_state.last_response.get("parameters", {})
        query = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        
        payload = {
            "query": query,
            "parameters": params,
            "domain": st.session_state.domain,
            "session_id": st.session_state.session_id
        }
        
        result = api_post("causal", payload, timeout=45)
        if result and "causal_results" in result:
            st.session_state.causal_analysis = result["causal_results"]
            st.session_state.active_tab = "causal"
            st.rerun()
        else:
            st.error("Causal analysis failed.")

def check_optimization_status():
    """Check optimization status"""
    with st.spinner("Checking optimization status..."):
        result = api_get(f"optimization/{st.session_state.session_id}", timeout=15)
        if result and "status" in result:
            st.session_state.optimization_status = result
            st.session_state.active_tab = "optimization"
            st.rerun()
        else:
            st.error("Failed to check optimization status.")

def run_arxiv_search():
    """Run arXiv search"""
    if not st.session_state.messages:
        st.warning("Please enter a query first.")
        return
    
    query = st.session_state.messages[-1]["content"]
    
    with st.spinner("Searching arXiv..."):
        payload = {"query": query}
        result = api_post("arxiv", payload, timeout=20)
        
        if result and "links" in result:
            st.session_state.arxiv_papers = result["links"]
            st.session_state.active_tab = "arxiv"
            st.rerun()
        else:
            st.error("arXiv search failed.")

# ==================== DASHBOARD ====================
def render_dashboard():
    """Render dashboard with metrics and insights"""
    st.markdown("### 📊 Performance Dashboard")
    
    if not st.session_state.get("last_response"):
        st.info("Run a query to see dashboard insights.")
        return
    
    resp = st.session_state.last_response
    
    # Create metrics cards
    st.markdown("#### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{resp.get('confidence', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            time_sec = resp.get('processing_time_seconds', 0)
            st.metric("Response Time", f"{time_sec:.1f}s")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            params = resp.get('parameters', {})
            st.metric("Parameters", len(params))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            reward = resp.get('reward_score', 0)
            st.metric("RLHF Reward", f"{reward:.3f}" if reward is not None else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Pipeline insights
    st.markdown("#### 🔬 Pipeline Insights")
    with st.expander("View Pipeline Details", expanded=False):
        if "trace" in resp:
            steps = resp["trace"]
            if steps:
                # Create step timeline
                step_names = [step.get("step", f"Step {i}").replace("_", " ").title() 
                            for i, step in enumerate(steps[:5])]
                step_times = [step.get("time_seconds", 0) for step in steps[:5]]
                
                if step_times:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=step_names,
                            y=step_times,
                            text=[f"{t:.1f}s" for t in step_times],
                            textposition='auto',
                            marker_color='#667eea'
                        )
                    ])
                    fig.update_layout(
                        title="Pipeline Step Durations",
                        xaxis_title="Step",
                        yaxis_title="Time (seconds)",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Parameter visualization
    if params:
        st.markdown("#### 📋 Parameter Summary")
        with st.expander("View All Parameters", expanded=False):
            display_parameters(params)

# ==================== ENHANCED SIDEBAR ====================
def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.markdown('<div class="main-title">IXORA</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Domain selection
    st.sidebar.markdown("#### 🎯 Research Domain")
    domain = st.sidebar.selectbox(
        "Select Domain",
        options=["biomed", "cs", "general"],
        index=["biomed", "cs", "general"].index(st.session_state.domain),
        key="domain_select",
        label_visibility="collapsed"
    )
    if domain != st.session_state.domain:
        st.session_state.domain = domain
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Connection test
    st.sidebar.markdown("#### 🔗 Connection")
    if st.sidebar.button("Test Connection", use_container_width=True, key="test_conn"):
        success, result = test_backend_connection()
        if success:
            st.sidebar.success("✅ Connected")
        else:
            st.sidebar.error(f"❌ {result}")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("#### 📱 Navigation")
    
    nav_options = [
        ("💬 Chat", "chat"),
        ("📊 Dashboard", "dashboard"),
        ("🔬 Causal", "causal"),
        ("📈 Optimization", "optimization"),
        ("📚 Literature", "arxiv")
    ]
    
    for option_text, option_value in nav_options:
        if st.sidebar.button(
            option_text,
            use_container_width=True,
            key=f"nav_{option_value}",
            type="primary" if st.session_state.active_tab == option_value else "secondary"
        ):
            st.session_state.active_tab = option_value
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display Preferences
    st.sidebar.markdown("#### 🎨 Display Preferences")
    
    if "show_trace_expanded" not in st.session_state:
        st.session_state.show_trace_expanded = False
    if "show_validation_expanded" not in st.session_state:
        st.session_state.show_validation_expanded = False
    
    st.session_state.show_trace_expanded = st.sidebar.checkbox(
        "📊 Expand Pipeline Trace by default",
        value=st.session_state.show_trace_expanded,
        key="trace_expand_toggle"
    )
    
    st.session_state.show_validation_expanded = st.sidebar.checkbox(
        "🔬 Expand Validation Metrics by default",
        value=st.session_state.show_validation_expanded,
        key="validation_expand_toggle"
    )
    
    st.sidebar.markdown("---")
    
    # Session management
    st.sidebar.markdown("#### ⚙️ Session")
    if st.sidebar.button("🔄 New Session", use_container_width=True):
        st.session_state.clear()
        init_session_state()
        st.rerun()
    
    # Session info
    st.sidebar.markdown("#### 📝 Session Info")
    st.sidebar.write(f"**ID:** `{st.session_state.session_id[:8]}...`")
    st.sidebar.write(f"**Domain:** `{st.session_state.domain}`")
    st.sidebar.write(f"**Messages:** `{len(st.session_state.messages)}`")
    
    # Debug info
    with st.sidebar.expander("🔍 Debug Info"):
        if st.session_state.get("last_response"):
            resp = st.session_state.last_response
            st.write(f"**Confidence:** {resp.get('confidence', 0):.2f}")
            st.write(f"**Time:** {resp.get('processing_time_seconds', 0):.2f}s")
            st.write(f"**Intent:** {resp.get('intent', 'unknown')}")
            st.write(f"**Pipeline:** {'Full' if resp.get('used_full_pipeline') else 'Fast'}")

# ==================== FEEDBACK HELPERS ====================
def hash_query(text: str) -> str:
    if not text:
        return "no-query"
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def send_feedback(session_id: str, preference: str, response_text: str,
                  query_text: str = "", reason: str = "") -> bool:
    """Synchronous feedback sender — works reliably in Streamlit"""
    query_hash = hash_query(query_text)
    
    payload = {
        "session_id": session_id,
        "preference": preference,
        "response": response_text[:1500],
        "query_hash": query_hash,
        "reason": reason[:200]
    }
    
    try:
        url = f"{BASE_API_URL}:{DOMAIN_PORTS.get(st.session_state.domain, 8000)}/feedback"
        resp = httpx.post(url, json=payload, timeout=6.0)
        if resp.status_code == 200:
            st.toast(f"Feedback recorded — thank you! ({preference})", icon="🙏")
            return True
        else:
            st.error(f"Feedback failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        st.error(f"Could not send feedback: {str(e)}")
        return False

# ==================== CHAT TAB WITH XML PARSING ====================

# ==================== CHAT TAB WITH XML PARSING ====================
def render_chat_tab():
    """Render the chat interface with XML parsing support"""
    # Action buttons
    render_action_buttons()
    st.markdown("---")
    
    # DEBUG: Show what's in the last response
    if st.session_state.get("last_response"):
        with st.expander("🔍 DEBUG: Raw API Response", expanded=False):
            st.json(st.session_state.last_response)
    
    # DEBUG: Show what's in messages
    if st.session_state.messages:
        with st.expander("🔍 DEBUG: Message Structure", expanded=False):
            for idx, msg in enumerate(st.session_state.messages):
                if msg["role"] == "assistant":
                    st.write(f"Message {idx} - Length: {len(msg.get('content', ''))}")
                    st.write(f"First 500 chars: {msg.get('content', '')[:500]}...")
                    if "confidence" in msg:
                        st.write(f"Confidence: {msg.get('confidence')}")
                    if "reward_score" in msg:
                        st.write(f"Reward Score: {msg.get('reward_score')}")
                    st.write("---")
    
    # Chat history display
    st.markdown("#### 💬 Conversation")
    
    for idx, msg in enumerate(st.session_state.messages):
        with st.container():
            # User message
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            
            # Assistant message with XML parsing
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🧪"):
                    # Check if response has XML structure
                    sections = parse_xml_response(msg["content"])
                    
                    # Debug: Show what was parsed
                    if st.session_state.get("debug_mode", False):
                        with st.expander("🔍 XML Debug", expanded=False):
                            st.write("Parsed sections:", sections)
                            st.write("Has explanation:", bool(sections.get("explanation")))
                    
                    # Render based on whether XML was found
                    if sections.get("explanation") or sections.get("enthusiasm"):
                        render_parsed_response(sections, st.session_state.domain)
                    else:
                        # Plain text response
                        st.markdown(msg["content"])
                    
                    # Metrics and metadata
                    if isinstance(msg, dict) and len(msg) > 1:
                        cols = st.columns(4)
                        
                        with cols[0]:
                            if "confidence" in msg:
                                conf = msg["confidence"]
                                cls = "high" if conf >= 0.85 else "medium" if conf >= 0.6 else "low"
                                st.markdown(f'<span class="confidence-badge confidence-{cls}">Confidence: {conf:.0%}</span>', 
                                           unsafe_allow_html=True)
                        
                        with cols[1]:
                            if "reward_score" in msg and msg["reward_score"] is not None:
                                score = float(msg["reward_score"])
                                cls = "high" if score > 0.1 else "medium" if score > -0.1 else "low"
                                st.markdown(f'<span class="confidence-badge confidence-{cls}">RLHF Reward: {score:.3f}</span>',
                                           unsafe_allow_html=True)
                            else:
                                st.caption("RLHF Reward: —")
                        
                        with cols[2]:
                            if "processing_time" in msg:
                                st.caption(f"⏱️ {msg['processing_time']:.1f}s")
                            elif "processing_time_seconds" in msg:
                                st.caption(f"⏱️ {msg['processing_time_seconds']:.1f}s")
                        
                        with cols[3]:
                            if "intent" in msg:
                                st.caption(f"🎯 {msg['intent']}")
                        
                        # Expandable sections
                        if msg.get("parameters"):
                            with st.expander("📋 Extracted Parameters", expanded=False):
                                display_parameters(msg["parameters"])
                        
                        # Display validation metrics for historical messages
                        if msg.get("embedding_scores") or msg.get("validation_scores"):
                            validation_data = {
                                "confidence": msg.get("confidence"),
                                "embedding_scores": msg.get("embedding_scores", {}),
                                "validation_scores": msg.get("validation_scores", {})
                            }
                            with st.expander("🔬 Validation Metrics", expanded=st.session_state.get("show_validation_expanded", False)):
                                display_validation_metrics(validation_data)
                        
                        if msg.get("trace"):
                            with st.expander("🔍 Pipeline Trace", expanded=st.session_state.get("show_trace_expanded", False)):
                                display_enhanced_trace(msg["trace"])
                        
                        # Feedback buttons - moved to separate container
                        if "content" in msg and msg["content"]:
                            prev_user_msg = ""
                            if idx > 0 and st.session_state.messages[idx-1]["role"] == "user":
                                prev_user_msg = st.session_state.messages[idx-1]["content"]
                            
                            st.markdown("---")
                            st.markdown("#### 💭 Was this response helpful?")
                            
                            fb_cols = st.columns([1, 1, 5])
                            with fb_cols[0]:
                                if st.button("👍 Helpful", key=f"like_{idx}_{hash(msg['content'][:20])}",
                                            use_container_width=True):
                                    send_feedback(
                                        session_id=st.session_state.session_id,
                                        preference="good",
                                        response_text=msg["content"],
                                        query_text=prev_user_msg
                                    )
                                    st.rerun()
                            
                            with fb_cols[1]:
                                if st.button("👎 Not helpful", key=f"dislike_{idx}_{hash(msg['content'][:20])}",
                                            use_container_width=True):
                                    with st.expander("Tell us why (optional)", expanded=True):
                                        reason = st.text_area("", placeholder="Too long / inaccurate / unsafe / ...", 
                                                             height=80, key=f"reason_{idx}")
                                        if st.button("Submit feedback", key=f"submit_reason_{idx}"):
                                            send_feedback(
                                                session_id=st.session_state.session_id,
                                                preference="bad",
                                                response_text=msg["content"],
                                                query_text=prev_user_msg,
                                                reason=reason
                                            )
                                            st.rerun()
    
    # New message input
    st.markdown("---")
    user_input = st.chat_input("Ask about experiments, parameters, hypotheses...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("🧪 Analyzing with multi-agent pipeline..."):
                payload = {
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "domain": st.session_state.domain
                }
                resp = api_post("chat", payload, timeout=220)
                
                if resp and "response" in resp:
                    # Create assistant message object
                    msg = {
                        "role": "assistant",
                        "content": resp["response"],
                        "confidence": resp.get("confidence", 0.75),
                        "reward_score": resp.get("reward_score"),
                        "trace": resp.get("trace", []),
                        "parameters": resp.get("parameters", {}),
                        "embedding_scores": resp.get("embedding_scores", {}),
                        "validation_scores": resp.get("validation_scores", {}),
                        "processing_time": resp.get("processing_time_seconds", 0),
                        "intent": resp.get("intent", "unknown"),
                        "used_full_pipeline": resp.get("used_full_pipeline", False)
                    }
                    st.session_state.messages.append(msg)
                    st.session_state.last_response = msg
                    st.session_state.detailed_trace = msg["trace"]
                    
                    # Parse and display with XML support
                    sections = parse_xml_response(msg["content"])
                    
                    if sections.get("explanation") or sections.get("enthusiasm"):
                        render_parsed_response(sections, st.session_state.domain)
                    else:
                        st.markdown(msg["content"])
                    
                    # Show metrics
                    cols = st.columns(4)
                    with cols[0]:
                        conf = msg["confidence"]
                        cls = "high" if conf >= 0.85 else "medium" if conf >= 0.6 else "low"
                        st.markdown(f'<span class="confidence-badge confidence-{cls}">Confidence: {conf:.0%}</span>', 
                                   unsafe_allow_html=True)
                    
                    with cols[1]:
                        if msg["reward_score"] is not None:
                            score = float(msg["reward_score"])
                            cls = "high" if score > 0.1 else "medium" if score > -0.1 else "low"
                            st.markdown(f'<span class="confidence-badge confidence-{cls}">RLHF Reward: {score:.3f}</span>',
                                       unsafe_allow_html=True)
                        else:
                            st.caption("RLHF Reward: —")
                    
                    with cols[2]:
                        st.caption(f"⏱️ {msg['processing_time']:.1f}s")
                    
                    with cols[3]:
                        st.caption(f"🎯 {msg['intent']}")
                    
                    if msg["parameters"]:
                        with st.expander("📋 Extracted Parameters", expanded=False):
                            display_parameters(msg["parameters"])
                    
                    # Display validation metrics if available
                    if resp.get("embedding_scores") or resp.get("validation_scores"):
                        validation_data = {
                            "confidence": msg.get("confidence"),
                            "embedding_scores": resp.get("embedding_scores", {}),
                            "validation_scores": resp.get("validation_scores", {})
                        }
                        with st.expander("🔬 Validation Metrics", expanded=st.session_state.get("show_validation_expanded", False)):
                            display_validation_metrics(validation_data)
                    
                    if msg["trace"]:
                        with st.expander("🔍 Pipeline Trace", expanded=st.session_state.get("show_trace_expanded", False)):
                            display_enhanced_trace(msg["trace"])
                    
                    if resp.get("optimization_note"):
                        st.info(f"📈 {resp['optimization_note']}")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to get response from server")
                    if resp and "error" in resp:
                        st.error(f"Error: {resp['error']}")


# ==================== MAIN CONTENT ====================
def render_main_content():
    """Render main content based on active tab"""
    
    # Main title
    st.markdown('<div class="main-title">🧪 IXORA</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
    
    # Tab-based content
    if st.session_state.active_tab == "chat":
        render_chat_tab()
    elif st.session_state.active_tab == "dashboard":
        render_dashboard()
    elif st.session_state.active_tab == "causal":
        display_causal_analysis()
    elif st.session_state.active_tab == "optimization":
        display_optimization_results()
    elif st.session_state.active_tab == "arxiv":
        display_arxiv_tab(st.session_state.arxiv_papers)

# ==================== MAIN APP ====================
def main():
    """Main app function"""
    init_session_state()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()