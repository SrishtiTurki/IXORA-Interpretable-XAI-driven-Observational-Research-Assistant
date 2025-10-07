import os
import json
import re
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import streamlit.components.v1 as components

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="🧪 AI Bio Research Assistant", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
}
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin-bottom: 1rem;
}
.suggestion-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
}
.xai-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
}
.causal-box {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
}
.technique-box {
    background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: white;
}
.flow-box {
    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: #333;
}
.chart-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.parameter-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    margin: 0.2rem;
    font-size: 0.8rem;
}
.assistant-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: white;
}
.user-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: white;
}
.status-success {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 0.5rem;
    border-radius: 5px;
    color: white;
    text-align: center;
}
.status-error {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 0.5rem;
    border-radius: 5px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧪 AI Biomedical Research Assistant</h1>
    <p>Powered by BioGPT + Mistral • SHAP/LIME Analysis • Parameter Optimization</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []
if "current_variables" not in st.session_state:
    st.session_state.current_variables = ["dose", "temperature"]
if "current_bounds" not in st.session_state:
    st.session_state.current_bounds = {
        "dose": [0.0, 1000.0],
        "temperature": [20.0, 40.0]
    }
if "auto_suggest" not in st.session_state:
    st.session_state.auto_suggest = True
if "enable_xai" not in st.session_state:
    st.session_state.enable_xai = True
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Research Configuration")
    
    # API Status with better checking
    st.subheader("📡 Connection Status")
    try:
        status_res = requests.get(f"{API_URL}/", timeout=10)
        if status_res.status_code == 200:
            st.markdown('<div class="status-success">✅ Backend Connected</div>', unsafe_allow_html=True)
            status_data = status_res.json()
            
            # Check model status
            try:
                model_res = requests.get(f"{API_URL}/debug/models", timeout=10)
                if model_res.status_code == 200:
                    model_data = model_res.json()
                    st.write("**AI Models Status:**")
                    st.write(f"• BioGPT: {'✅' if model_data.get('biogpt_loaded') else '❌'}")
                    st.write(f"• Mistral: {'✅' if model_data.get('mistral_loaded') else '❌'}")
            except:
                st.write("**AI Models:** Checking...")
                
        else:
            st.markdown('<div class="status-error">❌ Backend Issues</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown('<div class="status-error">❌ Backend Unreachable</div>', unsafe_allow_html=True)
        st.write(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Session management
    st.subheader("💾 Session")
    if st.session_state.session_id:
        st.write(f"**Session ID:** {st.session_state.session_id}")
        if st.button("🔄 New Session"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
    else:
        if st.button("🆕 Create Session"):
            st.session_state.session_id = str(np.random.randint(100000, 999999))
            st.rerun()
    
    st.markdown("---")
    
    # Auto-features toggle
    st.subheader("🤖 AI Features")
    st.session_state.auto_suggest = st.toggle("Auto-suggest optimizations", value=st.session_state.auto_suggest, help="Automatically run analysis on requests")
    st.session_state.enable_xai = st.toggle("Show detailed insights", value=st.session_state.enable_xai, help="Include parameter analysis and recommendations")
    
    st.markdown("---")
    
    # Current experimental setup
    st.subheader("⚗️ Experimental Variables")
    
    # Domain selection
    domain = st.selectbox("Research Domain", [
        "clinical_research", "medicine", "pharmacology", "molecular_biology",
        "biochemistry", "pathology", "immunology", "microbiology", "oncology", "neuroscience"
    ])
    
    # Quick variable presets
    preset = st.selectbox("Variable Preset", [
        "Drug Discovery", "Cell Culture", "Protein Analysis", "Genomics", "Custom"
    ])
    
    if preset == "Drug Discovery":
        st.session_state.current_variables = ["dose", "temperature", "pH"]
        st.session_state.current_bounds = {
            "dose": [0.0, 1000.0],
            "temperature": [20.0, 40.0],
            "pH": [6.0, 8.0]
        }
    elif preset == "Cell Culture":
        st.session_state.current_variables = ["serum_conc", "co2_level", "temperature"]
        st.session_state.current_bounds = {
            "serum_conc": [0.0, 20.0],
            "co2_level": [3.0, 10.0],
            "temperature": [35.0, 40.0]
        }
    elif preset == "Protein Analysis":
        st.session_state.current_variables = ["buffer_ph", "salt_conc", "incubation_time"]
        st.session_state.current_bounds = {
            "buffer_ph": [6.0, 9.0],
            "salt_conc": [0.0, 500.0],
            "incubation_time": [10.0, 180.0]
        }
    elif preset == "Genomics":
        st.session_state.current_variables = ["pcr_cycles", "annealing_temp", "primer_conc"]
        st.session_state.current_bounds = {
            "pcr_cycles": [20.0, 45.0],
            "annealing_temp": [50.0, 70.0],
            "primer_conc": [0.1, 2.0]
        }
    
    # Show current variables
    st.write("**Current Variables:**")
    for var in st.session_state.current_variables:
        bounds = st.session_state.current_bounds.get(var, [0.0, 100.0])
        st.write(f"• {var}: {bounds[0]:.1f} - {bounds[1]:.1f}")

# Main chat interface
st.subheader("💬 Research Assistant Chat")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show suggestions if available
            if message["role"] == "assistant" and "suggestions" in message and message["suggestions"]:
                with st.container():
                    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                    st.markdown("**💡 Suggestions:**")
                    for suggestion in message["suggestions"]:
                        st.write(f"• {suggestion}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Show parameters if available
            if message["role"] == "assistant" and "extracted_parameters" in message:
                params = message["extracted_parameters"]
                if params:
                    with st.container():
                        st.markdown('<div class="xai-box">', unsafe_allow_html=True)
                        st.markdown("**📊 Extracted Parameters:**")
                        for param, value in params.items():
                            st.markdown(f'<span class="parameter-badge">{param}: {value}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show analysis data if available
            if message["role"] == "assistant" and "analysis_data" in message and st.session_state.enable_xai:
                analysis = message["analysis_data"]
                if analysis and isinstance(analysis, dict) and len(analysis) > 0:
                    with st.container():
                        st.markdown('<div class="xai-box">', unsafe_allow_html=True)
                        st.markdown("**📈 Analysis Results:**")
                        
                        if "parameters" in analysis and analysis["parameters"]:
                            st.markdown("**Parameters Found:**")
                            for param, value in analysis["parameters"].items():
                                st.write(f"• {param}: {value}")
                        
                        if "parameter_count" in analysis:
                            st.metric("Parameters Count", analysis["parameter_count"])
                            
                        if "optimized_params" in analysis and analysis["optimized_params"]:
                            st.markdown("**Optimized Parameters:**")
                            for param, value in analysis["optimized_params"].items():
                                st.write(f"• {param}: {value}")
                                
                        if "optimized_outcome" in analysis:
                            st.metric("Optimized Outcome", f"{analysis['optimized_outcome']:.3f}")
                            
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show PubMed links if available
            if message["role"] == "assistant" and "paper_links" in message and message["paper_links"]:
                with st.container():
                    st.markdown('<div class="technique-box">', unsafe_allow_html=True)
                    st.markdown("**📚 Related Research Papers:**")
                    for i, link in enumerate(message["paper_links"]):
                        if link and link != "":
                            st.markdown(f"• [Paper {i+1}]({link})")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Show evaluation metrics if available
            if message["role"] == "assistant" and "evaluation" in message and st.session_state.enable_xai:
                evaluation = message["evaluation"]
                if evaluation and isinstance(evaluation, dict):
                    with st.container():
                        st.markdown('<div class="causal-box">', unsafe_allow_html=True)
                        st.markdown("**📊 Response Evaluation:**")
                        
                        relevance = evaluation.get('relevance_score', 0)
                        confidence = evaluation.get('confidence', 0)
                        
                        st.write(f"• Relevance Score: {relevance:.3f}")
                        st.write(f"• Confidence: {confidence:.3f}")
                        
                        notes = evaluation.get('notes', '')
                        if notes:
                            st.write(f"• Notes: {notes}")
                            
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show model usage info
            if message["role"] == "assistant":
                mistral_used = message.get("mistral_used", False)
                biogpt_used = message.get("biogpt_used", False)
                
                if mistral_used or biogpt_used:
                    with st.container():
                        st.markdown('<div class="flow-box">', unsafe_allow_html=True)
                        st.markdown("**🔧 Models Used:**")
                        if biogpt_used:
                            st.write("• BioGPT: Biomedical knowledge")
                        if mistral_used:
                            st.write("• Mistral: Response generation")
                        st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about experimental design, optimization, or analysis (e.g., 'Analyze dose=5mg at 37°C')..."):
    # Check if we have a session ID
    if not st.session_state.session_id:
        st.session_state.session_id = str(np.random.randint(100000, 999999))
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Determine which AI capabilities to invoke
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI (BioGPT + Mistral) is thinking..."):
            prompt_lower = prompt.lower()
            
            # Use chat endpoint for all requests
            endpoint = "/chat"
            
            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": st.session_state.session_id
            }
            
            try:
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    headers=headers,
                    json={"message": prompt},
                    timeout=180  # Increased timeout for complex analysis
                )
                
                if response.status_code != 200:
                    st.error(f"Backend error: {response.status_code}")
                    error_msg = f"Backend returned error {response.status_code}. Please try again."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
                
                data = response.json()
                
                # DEBUG: Show what we received
                # st.write("🔍 Debug - Response keys:", list(data.keys()))
                
                # FIXED: Handle multiple possible response field names
                reply = None
                possible_content_fields = ['response', 'content', 'analysis_response', 'result']
                
                for field in possible_content_fields:
                    if field in data and data[field]:
                        reply = data[field]
                        break
                
                if not reply:
                    # If no content found, try to get from nested structures
                    if 'result' in data and isinstance(data['result'], dict):
                        for field in possible_content_fields:
                            if field in data['result'] and data['result'][field]:
                                reply = data['result'][field]
                                break
                
                if not reply:
                    reply = "I received your message but couldn't generate a response. Please try again."
                
                # Display the response
                st.markdown(reply)
                
                # FIXED: Store with proper field mapping - handle multiple possible field names
                assistant_message = {
                    "role": "assistant", 
                    "content": reply,
                    "response_type": data.get("response_type", data.get("type", "general")),
                    "suggestions": data.get("suggestions", []),
                    "extracted_parameters": data.get("parameters_extracted", 
                                                   data.get("extracted_parameters", 
                                                          data.get("parameters_found", {}))),
                    "analysis_data": data.get("analysis_results", 
                                            data.get("technical_results",
                                                   data.get("data", {}))),
                    "paper_links": data.get("pubmed_references", 
                                          data.get("paper_links", [])),
                    "evaluation": data.get("evaluation", {}),
                    "mistral_used": data.get("mistral_used", False),
                    "biogpt_used": data.get("biogpt_used", False),
                    "analysis_performed": data.get("analysis_performed", False)
                }
                
                # Clean up empty fields
                assistant_message = {k: v for k, v in assistant_message.items() if v not in [None, "", [], {}]}
                
                st.session_state.messages.append(assistant_message)
                
            except requests.exceptions.Timeout:
                error_msg = "Request timeout - the server is taking too long to respond. Please try a simpler query or use /chat endpoint."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to the backend server. Please make sure the API is running on localhost:8000"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
            except Exception as e:
                error_msg = f"Service error: {str(e)}"
                st.error(error_msg)
                logger.error(f"API error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Quick action buttons
st.subheader("🚀 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🧪 Test Simple Query", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What is diabetes?"})
        st.rerun()

with col2:
    if st.button("📊 Test Analysis", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Analyze dose=50mg temperature=37°C"})
        st.rerun()

with col3:
    if st.button("🔬 PubMed Search", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Find recent papers about cancer immunotherapy"})
        st.rerun()

with col4:
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Session info section
with st.expander("📋 Session Information"):
    if st.session_state.session_id:
        try:
            session_response = requests.get(
                f"{API_URL}/session/{st.session_state.session_id}",
                timeout=10
            )
            if session_response.status_code == 200:
                session_data = session_response.json()
                st.write(f"**Session ID:** {session_data['session_id']}")
                st.write(f"**Message Count:** {session_data.get('message_count', 0)}")
                
                # FIX: Handle multiple field names for parameters
                extracted_params = (
                    session_data.get('extracted_parameters', {}) or 
                    session_data.get('parameters_extracted', {}) or
                    session_data.get('parameters_found', {})
                )
                if extracted_params:
                    st.write("**Extracted Parameters:**")
                    for param, value in extracted_params.items():
                        st.write(f"• {param}: {value}")
                
                st.write(f"**Biomedical Context:** {'Yes' if session_data.get('is_biomedical', False) else 'No'}")
            else:
                st.write("Could not retrieve session details from backend")
        except Exception as e:
            st.write(f"Session info unavailable: {str(e)}")
    else:
        st.write("No active session - start chatting to create a session!")

# Bottom status bar
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🧪 Experiments Logged", len(st.session_state.experiment_history))

with col2:
    st.metric("💬 Chat Messages", len(st.session_state.messages))

with col3:
    st.metric("📊 Variables Tracked", len(st.session_state.current_variables))

with col4:
    active_session = "Yes" if st.session_state.session_id else "No"
    st.metric("🔗 Active Session", active_session)

# Example prompts for new users
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 **Try these example queries:**")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.markdown("""
        **🔬 Parameter Analysis:**
        - *"Analyze dose=5mg at temperature=37°C"*
        - *"What's the effect of 50mg dose on cell viability?"*
        - *"Optimize pH and concentration for protein stability"*
        - *"Compare dose=10mg vs dose=20mg effects"*
        """)
    
    with example_col2:
        st.markdown("""
        **🧠 Biomedical Questions:**
        - *"Explain diabetes treatment mechanisms"*
        - *"What are recent advances in cancer immunotherapy?"*
        - *"How do antibiotics work against bacteria?"*
        - *"Tell me about Alzheimer's disease research"*
        """)

    st.markdown("### 🎯 **Example with Parameters:**")
    st.code("""
    "Analyze the effect of dose=50mg at temperature=37°C for 24 hours 
    with pH=7.4 on cell growth outcomes"
    """)

# Debug information (collapsed by default)
with st.expander("🐛 Debug Information"):
    st.write(f"**API URL:** {API_URL}")
    st.write(f"**Session ID:** {st.session_state.session_id}")
    
    # Enhanced backend status check
    try:
        status_res = requests.get(f"{API_URL}/", timeout=10)
        st.write(f"**Backend Status:** {status_res.status_code}")
        if status_res.status_code == 200:
            status_data = status_res.json()
            st.write(f"**Backend Version:** {status_data.get('version', 'Unknown')}")
            
            # Check health endpoint
            try:
                health_res = requests.get(f"{API_URL}/health", timeout=5)
                if health_res.status_code == 200:
                    health_data = health_res.json()
                    st.write("**Health Status:**")
                    for key, value in health_data.items():
                        st.write(f"• {key}: {value}")
            except:
                st.write("**Health Check:** Failed")
    except:
        st.write("**Backend Status:** Not reachable")
    
    st.write("**Recent Messages:**")
    for i, msg in enumerate(st.session_state.messages[-3:]):
        role = msg['role']
        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        st.write(f"{i+1}. {role}: {content_preview}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>🔬 AI Biomedical Research Assistant | Powered by BioGPT + Mistral | SHAP/LIME Analysis | Parameter Optimization</p>
    </div>
    """,
    unsafe_allow_html=True
)