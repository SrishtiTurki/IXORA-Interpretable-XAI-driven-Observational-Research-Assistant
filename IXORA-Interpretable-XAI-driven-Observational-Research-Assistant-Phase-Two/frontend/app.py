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
    "computerscience": int(os.getenv("CS_PORT", DEFAULT_PORT + 1)),
    "general": int(os.getenv("GENERAL_PORT", DEFAULT_PORT + 2))
}

# Set page configuration
st.set_page_config(
    page_title="� IXORA - AI Research Assistant",
    layout="wide",
    page_icon="�",
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
    """Initialize all session state variables with domain awareness"""
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
    
    # Domain-specific state
    if "domain_config" not in st.session_state:
        st.session_state.domain_config = {
            "biomed": {
                "icon": "🧪",
                "color": "#667eea",
                "title": "Biomedical Research"
            },
            "computerscience": {
                "icon": "💻",
                "color": "#9c27b0",
                "title": "Computer Science"
            },
            "general": {
                "icon": "🌐",
                "color": "#4caf50",
                "title": "General Research"
            }
        }
    
    # UI state
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
        
    if "domain_models" not in st.session_state:
        st.session_state.domain_models = {
            "biomed": "BioMistral-7B",
            "computerscience": "CodeQwen-7B",
            "general": "Mistral-7B"
        }


# ==================== HELPER FUNCTIONS ====================
def check_server_health(api_url):
    """Check if the server is online and return health status"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        return False, {"status": f"Error: {response.status_code}"}
    except Exception as e:
        return False, {"status": f"Connection error: {str(e)}"}

def extract_response_sections(response_text, domain="biomed"):
    """
    Extract structured sections from XML-like response
    
    Args:
        response_text: The raw response text containing XML tags
        domain: The domain context (biomed, computerscience, etc.)
    
    Returns:
        dict: Dictionary of extracted sections with their content
    """
    # Define domain-specific section tags
    domain_sections = {
        "biomed": ["enthusiasm", "explanation", "hypothesis", "followup"],
        "computerscience": ["technical", "algorithms", "implementation", "challenges", "followup"],
        "general": ["summary", "key_points", "followup"]
    }
    
    # Get sections for current domain or default to general
    sections = {section: "" for section in domain_sections.get(domain, domain_sections["general"])}
    
    # Simple XML parsing
    for section in sections.keys():
        start_tag = f"<{section}>"
        end_tag = f"</{section}>"
        
        if start_tag in response_text and end_tag in response_text:
            start = response_text.find(start_tag) + len(start_tag)
            end = response_text.find(end_tag)
            sections[section] = response_text[start:end].strip()
    
    # If no sections found with tags, return the full text in the first section
    if not any(sections.values()) and response_text.strip():
        first_section = next(iter(sections.keys()))
        sections[first_section] = response_text.strip()
    
    return sections

def display_assistant_response(response_text, confidence, domain="biomed"):
    """
    Display assistant response with domain-aware formatting
    
    Args:
        response_text: The assistant's response text
        confidence: Confidence score of the response (0-1)
        domain: The domain context (biomed, computerscience, etc.)
    """
    # Extract sections based on domain
    sections = extract_response_sections(response_text, domain)
    
    # Get domain-specific styling
    domain_cfg = st.session_state.domain_config.get(domain, {})
    domain_icon = domain_cfg.get("icon", "🤖")
    
    # Display confidence indicator
    confidence_color = "#4caf50" if confidence > 0.7 else "#ff9800" if confidence > 0.4 else "#f44336"
    st.caption(f"{domain_icon} Confidence: <span style='color:{confidence_color}'>{confidence*100:.1f}%</span>", unsafe_allow_html=True)
    
    # Domain-specific display logic
    if domain == "biomed":
        # Biomedical format: Enthusiasm, Explanation, Hypothesis, Followup
        if sections.get("enthusiasm"):
            st.markdown(f"💡 **{sections['enthusiasm']}**")
        
        if sections.get("explanation"):
            st.markdown(sections["explanation"])
        
        if sections.get("hypothesis"):
            with st.expander("🔬 Hypothesis"):
                st.markdown(sections["hypothesis"])
    
    elif domain == "computerscience":
        # Computer Science format: Technical, Algorithms, Implementation, Challenges, Followup
        if sections.get("technical"):
            st.markdown("### 🖥️ Technical Overview")
            st.markdown(sections["technical"])
        
        if sections.get("algorithms"):
            with st.expander("📊 Algorithms & Approaches"):
                st.markdown(sections["algorithms"])
        
        if sections.get("implementation"):
            with st.expander("⚙️ Implementation Details"):
                st.markdown(f"```python\n{sections['implementation']}\n```" if "```" in sections["implementation"] else sections["implementation"])
        
        if sections.get("challenges"):
            with st.expander("⚠️ Challenges & Considerations"):
                st.markdown(sections["challenges"])
    
    else:  # General/fallback format
        if sections.get("summary"):
            st.markdown(sections["summary"])
        
        if sections.get("key_points"):
            st.markdown("### Key Points")
            st.markdown(sections["key_points"])
    # Display follow-up questions if available (common across domains)
    if sections.get("followup"):
        st.markdown("---")
        st.markdown("### 🤔 Follow-up Questions")
        st.markdown(sections["followup"])
    
    # Fallback: If no sections were extracted, show the raw response
    if not any(sections.values()) and response_text.strip():
        st.markdown(response_text)
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
                    metrics_to_show = [
                        "confidence", 
                        "parameters_count", 
                        "explainability_method"
                    ]
                    
                    # Display each metric
                    for metric in metrics_to_show:
                        if metric in step_details:
                            st.metric(
                                metric.replace("_", " ").title(),
                                f"{step_details[metric]:.3f}" if isinstance(step_details[metric], (int, float)) else str(step_details[metric])
                            )

def render_sidebar():
    """Render the sidebar content with domain awareness"""
    with st.sidebar:
        # Get current domain config
        domain = st.session_state.get("selected_domain", "biomed")
        domain_cfg = st.session_state.domain_config.get(domain, {})
        
        # Dynamic title based on domain
        st.markdown(f"""
        <h1 style='text-align: center; color: {domain_cfg.get('color', '#667eea')}; margin-bottom: 0;'>{domain_cfg.get('icon', '🧠')} IXORA</h1>
        <p style='text-align: center; color: #666; margin-top: 0;'>{domain_cfg.get('title', 'AI Research Assistant')}</p>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Domain selection
        st.markdown("### 🔄 Research Domain")
        domain_options = [
            ("biomed", "🧪 Biomedical"),
            ("computerscience", "💻 Computer Science"),
            ("general", "🌐 General")
        ]
        
        selected = st.radio(
            "Select your research domain:",
            [opt[0] for opt in domain_options],
            format_func=lambda x: dict(domain_options)[x],
            index=[opt[0] for opt in domain_options].index(domain) if domain in [opt[0] for opt in domain_options] else 0,
            label_visibility="collapsed"
        )
        
        if selected != domain:
            st.session_state.selected_domain = selected
            st.rerun()
        
        # Model information
        st.markdown("---")
        st.markdown("### 🤖 Active Model")
        st.markdown(f"**{st.session_state.domain_models.get(domain, 'Mistral-7B')}**")
        
        # Domain-specific tips
        st.markdown("---")
        st.markdown("### 💡 Tips")
        if domain == "biomed":
            st.markdown("""
            - Ask about biomedical research, drug interactions, or clinical studies
            - Include specific parameters like dosage, pH, or temperature
            - Request explanations of medical terms or concepts
            """)
        elif domain == "computerscience":
            st.markdown("""
            - Ask about algorithms, code implementation, or system design
            - Include programming languages or frameworks
            - Request code examples or optimization tips
            """)
        else:
            st.markdown("""
            - Ask general research questions
            - Request explanations of complex topics
            - Get help with academic writing
            """)
        
        # Server status
        st.markdown("---")
        st.markdown("### 📡 Service Status")
        
        # Check server health for all domains
        cols = st.columns(2)
        for idx, (domain_name, port) in enumerate(DOMAIN_PORTS.items()):
            server_url = f"http://localhost:{port}"
            is_online, health = check_server_health(server_url)
            
            with cols[idx % 2]:
                domain_cfg = st.session_state.domain_config.get(domain_name, {})
                status = "🟢" if is_online else "🔴"
                st.metric(
                    f"{domain_cfg.get('icon', '')} {domain_name.title()}",
                    "Online" if is_online else "Offline",
                    status
                )
        
        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(f"""
        IXORA is an AI research assistant for {domain_cfg.get('title', 'scientific research')}.
        
        **Version:** 2.0.0  
        **Model:** {st.session_state.domain_models.get(domain, 'Mistral-7B')}
        **Domain:** {domain_cfg.get('title', 'General')}
        
        [GitHub Repository](https://github.com/yourusername/ixora) | 
        [Report an Issue](https://github.com/yourusername/ixora/issues)
        """)
        
        # Add some spacing at the bottom
        st.markdown("\n\n")

# ==================== MAIN UI ====================
def render_main_content(server_online, api_url):
    """Render the main content area with domain awareness"""
    # Get current domain config
    domain = st.session_state.get("selected_domain", "biomed")
    domain_cfg = st.session_state.domain_config.get(domain, {})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_response(message["content"], message.get("confidence", 0.9))
            else:
                st.markdown(message["content"])
    
    # Dynamic placeholder text based on domain
    placeholder_texts = {
        "biomed": "Ask me about biomedical research, drug interactions, or clinical studies...",
        "computerscience": "Ask about algorithms, code implementation, or system design...",
        "general": "Ask me anything about your research..."
    }
    
    # Chat input with domain-specific placeholder
    if prompt := st.chat_input(placeholder_texts.get(domain, "Ask me anything...")):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner(f"{domain_cfg.get('icon', '🤖')} Analyzing your {domain} query..."):
                try:
                    # Call the API with domain information
                    response = requests.post(
                        f"{api_url}/chat",
                        json={
                            "message": prompt,
                            "session_id": st.session_state.session_id,
                            "domain": domain  # Pass the current domain
                        },
                        timeout=90  # Increased timeout for complex queries
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Store the full response
                        st.session_state.last_response = data
                        
                        # Update extracted parameters if available
                        if "parameters" in data:
                            st.session_state.extracted_parameters = data["parameters"]
                        
                        # Update validation scores if available
                        if "validation_scores" in data:
                            st.session_state.validation_scores = data["validation_scores"]
                        
                        # Update trace data if available
                        if "trace" in data:
                            st.session_state.trace_data = data["trace"]
                        
                        # Display the response with domain-specific formatting
                        display_assistant_response(
                            data.get("response", "No response generated"),
                            data.get("confidence", 0.5)
                        )
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data.get("response", ""),
                            "confidence": data.get("confidence", 0.5),
                            "domain": domain
                        })
                        
                        # Show analysis tabs if we have data
                        if any(["parameters" in data, "validation_scores" in data, "trace" in data]):
                            st.session_state.show_analysis_tabs = True
                    
                    else:
                        error_msg = f"Error {response.status_code}: {response.text}"
                        st.error(error_msg)
                        logger.error(f"API Error: {error_msg}")
                
                except requests.exceptions.Timeout:
                    st.error("⚠️ The request timed out. The server might be busy. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Could not connect to the server. Please check your connection.")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
                    logger.exception("Error in render_main_content:")
        
        # Rerun to update the UI
        st.rerun()
    
    # Display analysis tabs if available
    if st.session_state.get("show_analysis_tabs", False):
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🔍 Parameters", "📈 Validation", "🔬 AI Trace"])
        
        with tab1:
            st.markdown("### 📊 Analysis")
            if st.session_state.get("last_response"):
                response = st.session_state.last_response
                if "analysis" in response:
                    st.json(response["analysis"], expanded=False)
                else:
                    st.info("No detailed analysis available for this response.")
        
        with tab2:
            st.markdown("### 🔍 Extracted Parameters")
            display_parameters_tab(st.session_state.get("extracted_parameters", {}))
        
        with tab3:
            st.markdown("### 📈 Response Validation")
            display_validation_tab(
                st.session_state.get("validation_scores", {}),
                st.session_state.last_response.get("confidence", 0.5) if "last_response" in st.session_state else 0.5
            )
        
        with tab4:
            st.markdown("### 🔬 AI Reasoning Trace")
            display_trace_tab(st.session_state.get("trace_data", []))
            
            # Show tools button if available
            if st.session_state.get("tools_visible", False):
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