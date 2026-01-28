# core/main.py - OPTIMIZED VERSION (< 180s target)
# Changes:
# 1. Causal analysis REMOVED from main flow (button-only via /causal endpoint)
# 2. Bayesian optimization moved to BACKGROUND (displays when ready)
# 3. Tighter parameter extraction with timeout

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
from typing import Optional, Dict, List, Any
import json
from datetime import datetime
import logging
import traceback
import torch
import uuid
import os
from fastapi import HTTPException
import hashlib
import numpy as np
from decimal import Decimal
import time

# modules
from core.langgraph import run_multi_agent
from core.analytics import run_causal_analysis, run_shap_analysis
from core.arxiv import retrieve_arxiv_evidence_async, _get_fallback_papers
from core.utils import cache_set, cache_get
from core.model_loader import get_model_status
from core.utils import detect_intent, load_session_state, save_session_state
from core.mistral import generate_with_mistral
from core.analytics import run_bayesian_optimization  # For background task

try:
    from core.rlhf.feedback_logger import log_feedback
except ImportError:
    try:
        from core.rlhf.feedback_logger import log_feedback_with_context as log_feedback
    except ImportError:
        def log_feedback(session_id, preference, response_text="", query_hash="unknown"):
            import logging
            logging.getLogger("biomed").info(f"Feedback logged (fallback): {preference}")
            return True

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for speed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend_debug.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("biomed")

app = FastAPI(title="IXORA - Multi-Agent Research Assistant (Optimized)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    domain: Optional[str] = "biomed"

class FeedbackItem(BaseModel):
    session_id: str
    preference: str
    response: str = ""
    query_hash: str = "unknown"

class CausalRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None
    include_links: bool = True
    domain: Optional[str] = "biomed"
    session_id: Optional[str] = None  # NEW: to fetch from session if needed

# ========== BACKGROUND TASK FOR BAYESIAN OPTIMIZATION ==========

# In main.py - Keep using the existing run_bayesian_optimization

async def run_optimization_background(session_id: str, parameters: Dict[str, Any], domain: str):
    """
    Background task that runs Bayesian optimization using EXISTING function.
    """
    logger.info(f"🔄 [BACKGROUND] Starting Bayesian optimization for session {session_id}")
    start_time = time.time()
    
    try:
        # Use the EXISTING function
        opt_result = await run_bayesian_optimization(parameters, domain=domain)
        
        # Load and update session state
        session_state = load_session_state(session_id) or {}
        
        session_state["bayesian_optimization"] = {
            "status": "completed",
            "result": opt_result,
            "duration": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "parameters_analyzed": list(parameters.keys())
        }
        
        save_session_state(session_id, session_state)
        
        logger.info(f"✅ [BACKGROUND] Optimization completed in {time.time() - start_time:.2f}s")
        logger.info(f"   Result status: {opt_result.get('status', 'unknown')}")
        
    except Exception as e:
        logger.error(f"❌ [BACKGROUND] Optimization failed: {e}")
        
        session_state = load_session_state(session_id) or {}
        session_state["bayesian_optimization"] = {
            "status": "failed",
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }
        save_session_state(session_id, session_state)
    
# ========== SESSION STATE MANAGEMENT ==========

def save_session_state(session_id: str, state: dict):
    """Save complete state for on-demand feature access"""
    cache_set(f"session:{session_id}", state)
    logger.info(f"Saved session state for {session_id}")

def load_session_state(session_id: str) -> dict:
    """Load session state"""
    state = cache_get(f"session:{session_id}")
    if state:
        logger.debug(f"Loaded session state for {session_id}")
    return state or {}

# ========== MAIN CHAT ENDPOINT (OPTIMIZED) ==========

# In main.py - Optimize the chat endpoint

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id or str(uuid.uuid4())
    domain = (req.domain or "biomed").lower()
    
    # Start timing
    start_time = time.time()
    logger.info(f"🚀 [START] Chat request | Session: {session_id} | Domain: {domain}")
    logger.info(f"📝 Query: {req.message[:100]}...")

    # Load/create session
    session_state = load_session_state(session_id) or {}
    history = session_state.get("history", [])
    history.append({"role": "user", "content": req.message})

    try:
        # Step 1: Quick intent detection (fast)
        intent_start = time.time()
        intent = await detect_intent(req.message, domain=domain)
        intent_time = time.time() - intent_start
        logger.info(f"🎯 Intent: {intent} ({intent_time:.2f}s)")

        # Handle non-research queries quickly
        if intent == "meta":
            quick_response = "Hi! I'm IXORA, your research assistant. I specialize in biomedical and computational research questions. What experiment or analysis are you working on?"
            return await _build_quick_response(session_id, session_state, history, quick_response, "meta")
        
        if intent == "explanatory":
            # Quick explanation
            explain_prompt = f"Provide a clear, concise explanation of: {req.message}"
            explanation, _ = await generate_with_mistral(explain_prompt, max_tokens=300, temperature=0.3)
            return await _build_quick_response(session_id, session_state, history, explanation, "explanatory")

        # Step 2: Run the main research pipeline
        logger.info("🔬 Starting research pipeline...")
        pipeline_start = time.time()
        
        result = await run_multi_agent(
            query=req.message,
            domain=domain,
            session_id=session_id,
            history=history
        )
        
        pipeline_time = time.time() - pipeline_start
        logger.info(f"✅ Pipeline completed in {pipeline_time:.2f}s")

        # Extract results
        response_text = result.get("final_response", "Response generation failed.")
        trace = result.get("trace", [])
        confidence = result.get("confidence", 0.7)
        white_box = result.get("white_box_state", {})
        raw_parameters = white_box.get("parameters", {})
        
        # Clean and standardize parameters
        parameters = {}
        for param_name, param_value in raw_parameters.items():
            if isinstance(param_value, dict):
                # Already a dictionary, use as-is but ensure it has required fields
                param_dict = {
                    "value": param_value.get("value", param_value.get("Value", "")),
                    "confidence": param_value.get("confidence", param_value.get("confidence_score", 0.8)),
                    "unit": param_value.get("unit", param_value.get("units", "")),
                    "raw_text": param_value.get("raw_text", param_value.get("original_text", str(param_value))),
                    "method": param_value.get("method", param_value.get("extraction_method", "extracted")),
                    "source": param_value.get("source", "pipeline")
                }
            else:
                # Simple value (string, number, etc.) - convert to standard format
                param_dict = {
                    "value": param_value,
                    "confidence": 0.8,
                    "unit": "",
                    "raw_text": str(param_value),
                    "method": "auto_detected",
                    "source": "pipeline"
                }
            
            # Clean up the parameter name
            clean_name = param_name.strip().replace(" ", "_").lower()
            parameters[clean_name] = param_dict

        # Step 3: Launch background optimization IF we have parameters
        optimization_launched = False
        if parameters:
            # Check if we have optimizable parameters
            has_optimizable_params = False
            for param_name, param in parameters.items():
                value = param.get("value", None)
                
                # Check if value is optimizable
                if isinstance(value, (int, float)):
                    has_optimizable_params = True
                    logger.debug(f"Found numeric parameter for optimization: {param_name} = {value}")
                    break
                elif isinstance(value, list) and len(value) == 2:
                    if all(isinstance(x, (int, float)) for x in value):
                        has_optimizable_params = True
                        logger.debug(f"Found range parameter for optimization: {param_name} = {value}")
                        break
                elif isinstance(value, str):
                    # Try to convert string to number
                    try:
                        float_val = float(value)
                        has_optimizable_params = True
                        logger.debug(f"Found string-to-numeric parameter for optimization: {param_name} = {value}")
                        break
                    except (ValueError, TypeError):
                        continue
            
            if has_optimizable_params:
                logger.info("⚡ Launching background optimization")
                # Ensure parameters are in the right format for optimization
                opt_parameters = {}
                for param_name, param in parameters.items():
                    if isinstance(param, dict) and "value" in param:
                        opt_parameters[param_name] = param["value"]
                    else:
                        opt_parameters[param_name] = param
                
                background_tasks.add_task(
                    run_optimization_background,
                    session_id,
                    opt_parameters,
                    domain
                )
                optimization_launched = True
            else:
                logger.info("ℹ️ No optimizable parameters found")

        # Step 4: Update session state
        history.append({"role": "assistant", "content": response_text})
        session_state.update({
            "history": history[-20:],
            "last_query": req.message,
            "last_response": response_text,
            "parameters": parameters,
            "trace": trace,
            "confidence": confidence,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "optimization_launched": optimization_launched,
            "pipeline_time": pipeline_time
        })
        
        # Store analytics if available
        if "analytics" in white_box:
            session_state["analytics"] = white_box["analytics"]
        
        save_session_state(session_id, session_state)

        # Step 5: Build final response
        total_time = time.time() - start_time
        
        response = {
            "response": response_text,
            "session_id": session_id,
            "confidence": round(confidence, 3),
            "trace": trace[:10],  # Limit trace size
            "parameters": parameters,  # Now properly formatted
            "processing_time_seconds": round(total_time, 2),
            "intent": intent,
            "domain": domain,
            "pipeline_stats": {
                "pipeline_time": round(pipeline_time, 2),
                "total_time": round(total_time, 2),
                "optimization_launched": optimization_launched,
                "parameters_extracted": len(parameters)
            },
            "analytics_available": "analytics" in white_box,
            "optimization_note": "Running in background - check /optimization/{session_id}" if optimization_launched else "No optimization needed"
        }

        # Add analytics summary if available
        if "analytics" in white_box:
            analytics = white_box["analytics"]
            response["analytics_summary"] = {
                "method_used": analytics.get("explainability_method", "none"),
                "has_explainability": bool(analytics.get("explainability")),
                "has_causal": bool(analytics.get("causal")),
                "parameter_count": len(parameters)
            }

        # Log completion
        logger.info(f"✅ [COMPLETE] Total time: {total_time:.2f}s")
        logger.info(f"   Parameters: {len(parameters)}")
        logger.info(f"   Confidence: {confidence:.2f}")
        logger.info(f"   Optimization launched: {optimization_launched}")
        logger.info("=" * 60)

        return response

    except asyncio.TimeoutError:
        logger.error(f"⏰ Request timed out for session {session_id}")
        return await _handle_timeout(session_id, session_state, history, req.message)
    
    except Exception as e:
        logger.exception(f"❌ Chat endpoint failed: {e}")
        return await _handle_error(session_id, session_state, history, e)
    

async def _build_quick_response(session_id, session_state, history, response_text, intent):
    """Build a quick response without full pipeline"""
    history.append({"role": "assistant", "content": response_text})
    session_state.update({
        "history": history[-20:],
        "intent": intent,
        "timestamp": datetime.now().isoformat()
    })
    save_session_state(session_id, session_state)
    
    return {
        "response": response_text,
        "session_id": session_id,
        "confidence": 0.9,
        "trace": [{"step": "quick_response", "intent": intent}],
        "parameters": {},
        "processing_time_seconds": 0.5,
        "intent": intent,
        "quick_response": True
    }

async def _handle_timeout(session_id, session_state, history, message):
    """Handle timeout errors"""
    error_msg = "⏰ The request timed out. Your query might be too complex or the server is busy. Please try a simpler query or try again later."
    history.append({"role": "assistant", "content": error_msg})
    session_state.update({
        "history": history[-20:],
        "error": "timeout",
        "timestamp": datetime.now().isoformat()
    })
    save_session_state(session_id, session_state)
    
    return {
        "response": error_msg,
        "session_id": session_id,
        "confidence": 0.3,
        "trace": [{"step": "error", "error": "timeout"}],
        "parameters": {},
        "processing_time_seconds": 60.0,
        "error": "timeout"
    }

async def _handle_error(session_id, session_state, history, error):
    """Handle general errors"""
    error_msg = f"❌ An error occurred: {str(error)[:100]}"
    history.append({"role": "assistant", "content": error_msg})
    session_state.update({
        "history": history[-20:],
        "error": str(error),
        "timestamp": datetime.now().isoformat()
    })
    save_session_state(session_id, session_state)
    
    return {
        "response": error_msg,
        "session_id": session_id,
        "confidence": 0.1,
        "trace": [{"step": "error", "error": str(error)}],
        "parameters": {},
        "processing_time_seconds": 0.0,
        "error": str(error)
    }

async def _build_quick_response(session_id, session_state, history, response_text, intent):
    """Build quick response for non-research queries"""
    history.append({"role": "assistant", "content": response_text})
    session_state.update({
        "history": history[-20:],
        "last_response": response_text,
        "timestamp": datetime.now().isoformat(),
        "intent": intent
    })
    save_session_state(session_id, session_state)
    
    return {
        "response": response_text,
        "session_id": session_id,
        "confidence": 0.9,
        "trace": [{"step": "quick_response", "intent": intent}],
        "parameters": {},
        "processing_time_seconds": 0.5,
        "intent": intent
    }


async def _handle_timeout(session_id, session_state, history, query):
    """Handle timeout gracefully"""
    fallback = f"I'm taking too long to analyze your query about '{query[:50]}...'. This might be a complex research question. Could you try rephrasing it or asking about a specific aspect?"
    
    history.append({"role": "assistant", "content": fallback})
    session_state["history"] = history[-20:]
    save_session_state(session_id, session_state)
    
    return {
        "response": fallback,
        "session_id": session_id,
        "confidence": 0.3,
        "trace": [{"step": "timeout", "error": "Request took too long"}],
        "parameters": {},
        "error": "Request timeout",
        "suggestion": "Try a more specific question or fewer parameters"
    }


async def _handle_error(session_id, session_state, history, error):
    """Handle error gracefully"""
    fallback = "I encountered an issue processing your research question. Please try rephrasing or ask about a specific experimental parameter."
    
    history.append({"role": "assistant", "content": fallback})
    session_state["history"] = history[-20:]
    save_session_state(session_id, session_state)
    
    return {
        "response": fallback,
        "session_id": session_id,
        "confidence": 0.1,
        "trace": [{"step": "error", "error": str(error)[:100]}],
        "parameters": {},
        "error": "Processing error"
    }

# In main.py - Add comprehensive logging helper

def log_pipeline_progress(step: str, duration: float, details: dict = None):
    """Log pipeline progress with timing"""
    logger.info(f"🔄 [{step.upper()}] Completed in {duration:.2f}s")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")

# Use it in the pipeline:
# log_pipeline_progress("extraction", extract_time, {"params": len(parameters)})

def _format_trace_summary(step: dict) -> str:
    """Format trace step into human-readable summary"""
    step_name = step.get("step", "unknown")
    
    if step_name == "extractor" or step_name == "parameter_extraction":
        count = step.get("param_count", 0)
        method = step.get("method", "unknown")
        return f"Extracted {count} parameters using {method}"
    
    elif step_name == "draft":
        length = step.get("draft_length", 0)
        return f"Generated draft response ({length} chars)"
    
    elif step_name == "analytics":
        methods = step.get("methods_used", [])
        return f"Ran analytics: {', '.join(methods)}" if methods else "Analytics completed"
    
    elif step_name == "hypothesis":
        return f"Formulated hypothesis: {step.get('hypothesis', '')[:100]}..."
    
    elif step_name == "synthesizer":
        return "Synthesized final response with evidence"
    
    elif step_name == "validator" or step_name == "validator_comprehensive":
        conf = step.get("confidence", 0)
        return f"Validated response (confidence: {conf:.2f})"
    
    else:
        return step.get("summary", f"Completed {step_name}")


# ========== CAUSAL ANALYSIS ENDPOINT (BUTTON-TRIGGERED) ==========

@app.post("/causal")
async def causal_analysis_endpoint(req: CausalRequest):
    """
    On-demand causal analysis triggered by frontend button.
    Can use parameters from session or from request.
    """
    logger.info(f"🔬 Causal analysis requested for: '{req.query[:100]}...'")
    
    try:
        # Get parameters (from request or session)
        parameters = req.parameters
        
        if not parameters and req.session_id:
            # Try to load from session
            session_state = load_session_state(req.session_id)
            parameters = session_state.get("parameters", {})
        
        if not parameters:
            return {
                "status": "error",
                "error": "No parameters available for causal analysis. Please run a query first.",
                "causal_results": {}
            }
        
        # Run causal analysis with timeout
        causal_result = await asyncio.wait_for(
            run_causal_analysis(parameters, domain=req.domain),
            timeout=30.0  # 30 second timeout
        )
        
        # Optionally fetch arXiv links if requested
        arxiv_links = []
        if req.include_links:
            try:
                arxiv_links = await asyncio.wait_for(
                    retrieve_arxiv_evidence_async(req.query, max_papers=3),
                    timeout=10.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"arXiv fetch failed: {e}")
                arxiv_links = _get_fallback_papers(req.query)
        
        return {
            "status": "success",
            "causal_results": causal_result,
            "arxiv_links": arxiv_links,
            "parameters_analyzed": list(parameters.keys()),
            "domain": req.domain
        }
        
    except asyncio.TimeoutError:
        logger.error("Causal analysis timed out")
        return {
            "status": "timeout",
            "error": "Causal analysis took too long (>30s). Please try with fewer parameters.",
            "causal_results": {}
        }
    
    except Exception as e:
        logger.exception(f"Causal analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e)[:200],
            "causal_results": {}
        }


# ========== OPTIMIZATION STATUS ENDPOINT ==========

@app.get("/optimization/{session_id}")
async def get_optimization_status(session_id: str):
    """
    Check if Bayesian optimization has completed for a session.
    Frontend can poll this endpoint to show results when ready.
    """
    try:
        session_state = load_session_state(session_id)
        
        if not session_state:
            raise HTTPException(status_code=404, detail="Session not found")
        
        opt_data = session_state.get("bayesian_optimization", {})
        
        if not opt_data:
            return {
                "status": "not_started",
                "message": "No optimization has been requested for this session"
            }
        
        return {
            "status": opt_data.get("status", "unknown"),
            "result": opt_data.get("result", {}),
            "timestamp": opt_data.get("timestamp", ""),
            "error": opt_data.get("error", None)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve optimization status")


# ========== ARXIV ENDPOINT ==========

@app.post("/arxiv")
async def arxiv_search(query: str):
    """Search for relevant arXiv papers"""
    if not query or len(query.strip()) < 3:
        return {
            "links": [],
            "error": "Query too short",
            "status": "error"
        }
    
    try:
        # Try arXiv API with timeout
        try:
            papers = await asyncio.wait_for(
                retrieve_arxiv_evidence_async(query, max_papers=5),
                timeout=15.0
            )
            
            if papers:
                logger.info(f"📚 Found {len(papers)} arXiv papers")
                return {
                    "links": papers,
                    "count": len(papers),
                    "status": "success",
                    "source": "arxiv_api"
                }
        
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"arXiv API failed: {e}, using fallback")
        
        # Use fallback papers
        fallback_papers = _get_fallback_papers(query)
        logger.info(f"📚 Using {len(fallback_papers)} fallback papers")
        
        return {
            "links": fallback_papers,
            "count": len(fallback_papers),
            "status": "partial",
            "source": "fallback",
            "note": "arXiv API unavailable, showing relevant reference papers"
        }
        
    except Exception as e:
        logger.error(f"arXiv endpoint error: {e}")
        return {
            "links": [],
            "error": "Server error",
            "status": "error"
        }


# ========== HEALTH & DIAGNOSTICS ==========

@app.get("/health")
async def health():
    """Health check with optimization info"""
    return {
        "status": "healthy",
        "service": "IXORA Research Assistant",
        "timestamp": datetime.now().isoformat(),
        "optimizations": {
            "parameter_extraction_timeout": "15s",
            "analytics_timeout": "30s",
            "pipeline_target": "< 90s",
            "bayesian_optimization": "background_task",
            "causal_analysis": "button_triggered",
            "features": [
                "Fast parameter extraction",
                "Essential analytics only",
                "Background optimization",
                "On-demand causal analysis"
            ]
        },
        "endpoints": {
            "/chat": "Main research pipeline",
            "/causal": "On-demand causal analysis",
            "/optimization/{session_id}": "Check background optimization status",
            "/arxiv": "Literature search"
        }
    }

@app.post("/feedback")
async def receive_feedback(item: FeedbackItem):
    """Receive user feedback (thumbs up/down) for RLHF training"""
    if item.preference not in ["good", "bad"]:
        raise HTTPException(status_code=400, detail="Preference must be 'good' or 'bad'")

    try:
        success = log_feedback(
            session_id=item.session_id,
            preference=item.preference,
            response_text=item.response,
            query_hash=item.query_hash
        )
        
        if success:
            return {"status": "success", "message": "Feedback recorded. Thank you!"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
            
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process feedback")

# ========== DEBUG ENDPOINTS ==========

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Debug endpoint to view session state"""
    state = load_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove large fields for cleaner display
    safe_state = {}
    for key, value in state.items():
        if key == "history":
            safe_state[key] = [{"role": msg.get("role", ""), "content": msg.get("content", "")[:100] + "..."} 
                              for msg in value[:5]]
        elif isinstance(value, str) and len(value) > 500:
            safe_state[key] = value[:500] + "..."
        elif isinstance(value, dict) and key == "parameters":
            safe_state[key] = {k: {"value": v.get("value", ""), "unit": v.get("unit", "")} 
                              for k, v in list(value.items())[:10]}
        else:
            safe_state[key] = value
    
    return safe_state

@app.get("/trace/{session_id}")
async def get_trace(session_id: str):
    """Get trace for a session"""
    state = load_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    trace = state.get("trace", [])
    return {
        "session_id": session_id,
        "trace_count": len(trace),
        "trace": trace[:20]
    }

@app.get("/models/status")
async def get_models_status():
    """Check which models are preloaded"""
    status = get_model_status()
    
    # Check if BioMistral is actually loaded and responsive
    biomistral_status = status.get("biomistral", {})
    if biomistral_status.get("loaded"):
        try:
            from core.model_loader import model_loader
            test_result = await model_loader.generate_with_biomistral("Test", max_tokens=5)
            biomistral_status["responsive"] = True
            biomistral_status["test_output"] = test_result[:50]
        except Exception as e:
            biomistral_status["responsive"] = False
            biomistral_status["error"] = str(e)
    
    return {
        "status": "ok",
        "models": status,
        "timestamp": datetime.now().isoformat(),
        "recommendation": "BioMistral should show 'loaded: true' and 'responsive: true' for optimal performance"
    }

# ========== STARTUP ==========

@app.on_event("startup")
async def startup():
    logger.info("="*80)
    logger.info("🚀 IXORA - Starting up (OPTIMIZED VERSION)")
    logger.info("="*80)
    
    import sys
    is_dev = "--reload" in sys.argv or os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    
    try:
        from core.model_loader import startup_models, model_loader
        
        if is_dev:
            logger.info("⚡ Dev mode: Fast startup + background model loading")
            asyncio.create_task(
                startup_models(domain="biomed", warmup=False)
            )
            logger.info("🔄 Heavy models loading in background...")
            logger.info("✅ Server ready immediately — models will be ready in ~20-40s")
        else:
            logger.info("🛡️ Production mode: Full pre-loading")
            await startup_models("biomed", warmup=True)
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        
        # Initialize reward model
        try:
            from core.rlhf.reward_model import get_reward_model
            reward_model = get_reward_model()
            if not reward_model._model_loaded:
                logger.info("📝 Initializing new reward model")
                os.makedirs("models", exist_ok=True)
                torch.save(reward_model.state_dict(), "models/reward_model.pth")
                reward_model._model_loaded = True
        except Exception as e:
            logger.warning(f"Reward model init warning: {e}")
        
        status = get_model_status()
        loaded = sum(1 for s in status.values() if s["loaded"])
        total = len(status)
        
        logger.info(f"📊 Models: {loaded}/{total} loaded successfully")
        for name, info in status.items():
            symbol = "✅" if info["loaded"] else "🔄" if info["loading"] else "❌"
            logger.info(f"   {symbol} {name}: {'Ready' if info['loaded'] else 'Loading' if info['loading'] else 'Failed'}")
        
        logger.info("✅ System ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        logger.warning("⚠️ Continuing with lazy loading mode")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")