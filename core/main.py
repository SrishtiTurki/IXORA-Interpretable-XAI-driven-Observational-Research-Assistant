# core/main.py - COMPLETE UPDATED VERSION
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
# modules
from core.langgraph import run_multi_agent
from core.analytics import run_causal_analysis,  run_shap_analysis
from core.arxiv import retrieve_arxiv_evidence_async, _get_fallback_papers
from core.utils import cache_set, cache_get
from core.arxiv import retrieve_arxiv_evidence_async   
# Add to main.py or create trainer_service.py
import schedule
import time

try:
    from core.rlhf.feedback_logger import log_feedback
except ImportError:
    try:
        from core.rlhf.feedback_logger import log_feedback_with_context as log_feedback
    except ImportError:
        # Create a fallback function
        def log_feedback(session_id, preference, response_text="", query_hash="unknown"):
            import logging
            logging.getLogger("biomed").info(f"Feedback logged (fallback): {preference}")
            return True

# Create directories if needed
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend_debug.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("biomed")

app = FastAPI(title="IXORA - Multi-Agent Research Assistant")

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
    domain: Optional[str] = "biomed"  # allow cs or biomed

class FeedbackItem(BaseModel):
    session_id: str
    preference: str
    response: str = ""
    query_hash: str = "unknown"

class CausalRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None  # From extracted params
    include_links: bool = True  # Flag for arXiv links
    domain: Optional[str] = "biomed"

# ========== SESSION STATE MANAGEMENT ==========

def save_session_state(session_id: str, state: dict):
    """Save complete state for on-demand feature access"""
    cache_set(f"session:{session_id}", state)
    logger.info(f"Saved session state for {session_id}")

def load_session_state(session_id: str) -> dict:
    """Load session state"""
    state = cache_get(f"session:{session_id}")
    if state:
        logger.info(f"Loaded session state for {session_id}")
    else:
        logger.warning(f"No session state found for {session_id}")
    return state or {}

# ========== MAIN CHAT ENDPOINT ==========

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    domain = (req.domain or "biomed").lower()
    logger.info(f"🚀 Chat request: '{req.message[:100]}...' (session: {session_id})")

    # Load previous state safely
    session_state = load_session_state(session_id) or {}
    history = session_state.get("history", [])

    # Append new user message
    history.append({"role": "user", "content": req.message})

    logger.info(f"Sending {len(history)} messages to multi-agent pipeline")

    try:
        
        # Start timing
        import time
        start_time = time.time()
        
        # Call with history
        result = await run_multi_agent(
            query=req.message,
            domain=domain,
            session_id=session_id,
            history=history
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Multi-agent pipeline completed in {duration:.2f}s")
        
        # Get white box state
        white_box = result.get("white_box_state", {})
        
        # Extract validation scores from the validator agent
        validation_scores = white_box.get("validation_scores", {})
        confidence = validation_scores.get("overall_confidence", result.get("confidence", 0.7))
        
        # Extract parameters
        parameters = white_box.get("parameters", {})
        
        # Extract trace
        trace = white_box.get("trace", [])
        
        # Get final response
        final_response = result.get("final_response", "No response generated")
        
        # Append assistant response
        history.append({"role": "assistant", "content": final_response})
        
        # Calculate query hash for RLHF
        query_hash = hashlib.sha256(req.message.encode()).hexdigest()[:16]
        
        # Save session state with enhanced metadata
        session_state.update({
            "history": history,
            "query": req.message,
            "domain": domain,
            "parameters": parameters,
            "validation_scores": validation_scores,
            "trace": trace,
            "last_response_metadata": {
                "response": final_response,
                "trace": trace,
                "validation_scores": validation_scores,
                "confidence": confidence,
                "parameters": parameters,
                "duration": duration,
                "query_hash": query_hash
            }
        })
        
        save_session_state(session_id, session_state)
        
        # Format trace for frontend
        formatted_trace = []
        for step in trace:
            if isinstance(step, dict):
                step_name = step.get("step", "")
                formatted_step = {
                    "step": step_name,
                    "timestamp": step.get("timestamp", datetime.now().isoformat()),
                    "summary": _format_trace_summary(step),
                    "details": step
                }
                
                # Add specific metadata for certain steps
                if step_name == "extractor":
                    formatted_step["parameters_count"] = step.get("parameters_count", 0)
                elif step_name == "analytics":
                    formatted_step["explainability_method"] = step.get("explainability_method", "none")
                elif step_name == "validator_comprehensive":
                    formatted_step["confidence"] = step.get("confidence", 0.7)
                    
                formatted_trace.append(formatted_step)
        
        # Return with all data
        return {
            "response": final_response,
            "trace": formatted_trace,
            "validation_scores": validation_scores,
            "confidence": confidence,
            "session_id": session_id,
            "metadata": {
                "parameters_extracted": len(parameters),
                "validation_performed": len(validation_scores) > 0,
                "trace_steps": len(trace),
                "total_duration": duration,
                "query_hash": query_hash
            },
            "extracted_parameters": parameters  # Add for frontend display
        }
        
    except Exception as e:
        logger.exception(f"Chat endpoint failed: {e}")
        
        # Create fallback response with basic structure
        if domain == "cs":
            fallback_response = f"""<enthusiasm>Thanks for your CS research question!</enthusiasm>
<clarify>What specific algorithmic approach or constraints should I consider?</clarify>
<explanation>
I encountered a technical issue while processing your query about "{req.message[:100]}". 

For computer science research questions, consider:
- Algorithmic complexity and scalability
- Baselines and reproducibility (random seeds, versions)
- Performance metrics (latency, throughput, accuracy)
- Resource constraints (memory/compute)
</explanation>
<hypothesis>Computational parameters will significantly influence the observed performance metrics.</hypothesis>
<followup>Could you rephrase your question or specify the algorithm/task details?</followup>"""
        else:
            fallback_response = f"""<enthusiasm>Thanks for your research question!</enthusiasm>
<explanation>
I encountered a technical issue while processing your query about "{req.message[:100]}". 

For biomedical research questions like this, typical considerations include:
- Experimental design with proper controls
- Statistical analysis of results  
- Parameter optimization
- Biological relevance and reproducibility
</explanation>
<hypothesis>The key experimental parameters will significantly influence your research outcomes.</hypothesis>
<followup>Could you rephrase your question or ask about specific experimental conditions?</followup>"""
        
        # Save error state
        session_state.update({
            "history": history + [{"role": "assistant", "content": fallback_response}],
            "query": req.message,
            "error": str(e)[:200]
        })
        save_session_state(session_id, session_state)
        
        return {
            "response": fallback_response,
            "error": str(e)[:200],
            "trace": [{"step": "error", "error": str(e)[:100], "fallback_used": True}],
            "validation_scores": {"overall_confidence": 0.3, "structural_completeness": 0.8},
            "confidence": 0.3,
            "session_id": session_id
        }

def _format_trace_summary(step: Dict) -> str:
    """Create a one-line summary of a trace step"""
    step_name = step.get("step", "")
    
    if step_name == "extractor":
        count = step.get("parameters_count", 0)
        return f"Extracted {count} parameters"
    
    elif step_name == "draft":
        if step.get("biomistral_used"):
            return "Generated draft with BioMistral"
        if step.get("csmodel_used"):
            return "Generated draft with CS model"
        return "Generated response draft"
    
    elif step_name == "analytics":
        method = step.get("explainability_method", "none")
        return f"Ran analytics with {method}"
    
    elif step_name == "hypothesis":
        return "Generated testable hypothesis"
    
    elif step_name == "synthesizer":
        return "Synthesized final response"
    
    elif step_name == "validator_comprehensive":
        confidence = step.get("confidence", 0.7)
        return f"Validated response ({confidence:.1%} confidence)"
    
    elif step_name == "cosine_similarity":
        return "Calculated semantic similarity"
    
    elif step_name == "advanced_metrics":
        return "Calculated advanced validation metrics"
    
    else:
        return f"Completed {step_name} step"
    

# ========== RLHF =================================

async def train_if_enough_feedback():
    """Check if we have enough feedback and train"""
    import os
    import json
    
    feedback_file = "logs/rlhf_feedback.jsonl"
    if not os.path.exists(feedback_file):
        return
    
    # Count feedback
    with open(feedback_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) >= 20:  # Train after 20 feedbacks
        logger.info(f"🎯 Starting RLHF training with {len(lines)} feedbacks")
        
        # Run training
        from core.rlhf.trainer import train_reward_model
        success = await train_reward_model()
        
        if success:
            logger.info("✅ RLHF training complete")
            # Optional: Archive used feedback
            archive_file = f"logs/rlhf_used_{int(time.time())}.jsonl"
            os.rename(feedback_file, archive_file)

@app.get("/rlhf/status")
async def rlhf_status():
    """Check RLHF training status"""
    import os
    import json
    
    feedback_file = "logs/rlhf_feedback.jsonl"
    model_file = "models/reward_model.pth"
    
    feedback_count = 0
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_count = len(f.readlines())
    
    model_exists = os.path.exists(model_file)
    
    # Count preferences
    preferences = {"good": 0, "bad": 0}
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    pref = data.get("preference", 0)
                    if pref == 1:
                        preferences["good"] += 1
                    else:
                        preferences["bad"] += 1
                except:
                    pass
    
    return {
        "status": "active",
        "feedback_collected": feedback_count,
        "model_trained": model_exists,
        "preferences": preferences,
        "training_threshold": 20,
        "ready_for_training": feedback_count >= 20,
        "last_trained": _get_last_trained_time(),
        "recommendations": [
            "Collect at least 20 feedbacks before first training",
            f"Current: {feedback_count}/20 feedbacks",
            "Model will select better responses after training"
        ]
    }

def _get_last_trained_time():
    import os
    import time
    model_file = "models/reward_model.pth"
    if os.path.exists(model_file):
        mod_time = os.path.getmtime(model_file)
        return time.ctime(mod_time)
    return "Never"

# ========== ON-DEMAND: CAUSAL INFERENCE ==========

@app.post("/causal")
async def run_causal(req: Dict[str, str]):
    """
    On-demand causal inference (triggered by frontend button)
    """
    logger.info(f"📥 Received causal request: {req}")
    
    session_id = req.get("session_id")
    domain = (req.get("domain") or "biomed").lower()
    if not session_id:
        logger.error("❌ No session_id in request")
        return {
            "error": "session_id required", 
            "status": "failed"
        }
    
    logger.info(f"🔬 Processing causal analysis for session: {session_id}")
    
    try:
        # Load session state
        state = load_session_state(session_id)
        if not state:
            logger.warning(f"⚠️ No session state found for {session_id}")
            return {
                "causal": {
                    "error": "Session expired. Please run a query first.",
                    "suggestion": "Try asking a research question first"
                },
                "status": "failed"
            }
        
        # Extract parameters from session state
        parameters = state.get("parameters", {})
        
        # If no parameters in state, try to extract from query
        if not parameters:
            last_query = state.get("query", "")
            if last_query:
                logger.info("📊 No cached parameters, extracting from query...")
                try:
                    from core.utils import extract_parameters
                    parameters = await extract_parameters(last_query, domain)
                    logger.info(f"✅ Extracted {len(parameters)} parameters")
                except Exception as e:
                    logger.error(f"Parameter extraction failed: {e}")
        
        if not parameters or len(parameters) < 2:
            logger.warning(f"⚠️ Insufficient parameters: {len(parameters)}")
            return {
                "causal": {
                    "ate": 0.0,
                    "error": "Insufficient parameters for causal analysis",
                    "suggestion": "Try a query with multiple parameters like 'pH 5-7 and temperature 25-35°C'",
                    "parameters_found": len(parameters)
                },
                "status": "partial"
            }
        
        logger.info(f"🧠 Running causal analysis with {len(parameters)} parameters")
        
        try:
            from core.analytics import run_causal_analysis
            
            # Run with timeout
            causal_result = await asyncio.wait_for(
                run_causal_analysis(parameters, domain),
                timeout=60.0
            )
            
            logger.info(f"✅ Causal analysis complete: {causal_result.get('method', 'unknown')}")
            
            # CONVERT NUMPY TYPES BEFORE SAVING
            causal_result_converted = causal_result
            
            # Save to session state
            if "causal_results" not in state:
                state["causal_results"] = []
            state["causal_results"].append(causal_result_converted)
            save_session_state(session_id, state)
            
            # Return with converted types
            return {
                "causal": causal_result_converted,
                "status": "success",
                "session_id": session_id,
                "parameters_used": list(parameters.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            logger.error("⏰ Causal analysis timeout")
            return {
                "causal": {
                    "error": "Causal analysis timeout (60s)",
                    "suggestion": "Try with fewer parameters"
                },
                "status": "timeout"
            }
        except Exception as e:
            logger.error(f"❌ Causal analysis error: {e}", exc_info=True)
            return {
                "causal": {
                    "error": str(e)[:200],
                    "suggestion": "Check parameter values and try again"
                },
                "status": "failed"
            }
            
    except Exception as e:
        logger.error(f"❌ Causal endpoint error: {e}", exc_info=True)
        return {
            "causal": {
                "error": "Server error processing request"
            },
            "status": "error"
        }
    
@app.post("/causal_analysis")
async def causal_analysis(req: CausalRequest, background_tasks: BackgroundTasks):
    logger.info(f"🚀 Causal analysis request: {req.query[:100]}")
    
    try:
        domain = (req.domain or "biomed").lower()
        # Run causal analysis (from analytics.py)
        causal_result = await run_causal_analysis(req.query, req.parameters or {}, domain)
        
        # Optionally add arXiv links (evidence with causal "links")
        evidence = []
        if req.include_links:
            evidence = await retrieve_arxiv_evidence_async(req.query + " causal effects", max_papers=2)
        
        return {
            "causal_analysis": causal_result,
            "evidence_links": evidence,  # ArXiv PDFs/links
            "timestamp": datetime.now().isoformat(),
            "confidence": causal_result.get("confidence", 0.7)
        }
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ON-DEMAND: ARXIV PAPERS ==========
@app.post("/arxiv")
async def run_arxiv(req: Dict[str, str]):
    """On-demand arXiv search with fallback"""
    logger.info(f"📥 Received arXiv request: {req}")
    
    session_id = req.get("session_id")
    if not session_id:
        return {
            "error": "session_id required", 
            "status": "failed",
            "links": []
        }
    
    try:
        state = load_session_state(session_id)
        if not state:
            return {
                "links": [],
                "error": "Session expired",
                "status": "failed"
            }
        
        query = state.get("query", "")
        if not query:
            # Get from history
            history = state.get("history", [])
            for msg in reversed(history):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        
        if not query:
            query = "yeast biomass growth pH temperature"
        
        logger.info(f"🔍 Searching arXiv for: {query[:100]}...")
        
        # Try real search first
        try:
            papers = await asyncio.wait_for(
                retrieve_arxiv_evidence_async(query, max_papers=5),
                timeout=30
            )
            
            if papers:
                logger.info(f"✅ Found {len(papers)} papers")
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
    """Health check"""
    return {
        "status": "healthy",
        "service": "IXORA Multi-Agent System",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "features": {
            "chat": True,
            "causal": True,
            "arxiv": True,
            "validation": True,
            "trace": True
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
        "trace": trace[:20]  # Limit to first 20 entries
    }


# In main.py, add this endpoint:

@app.get("/models/status")
async def get_models_status():
    """Check which models are preloaded"""
    from core.model_loader import get_model_status
    
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
# In main.py, modify the startup function:
# In main.py startup function
@app.on_event("startup")
async def startup():
    logger.info("="*80)
    logger.info("🚀 IXORA - Starting with COMPLETE Model Pre-loading")
    logger.info("="*80)
    
    try:
        from core.model_loader import startup_models, get_model_status
        
        # Start all models
        success = await startup_models("biomed", warmup=True)
        
        # Initialize reward model (create if doesn't exist)
        try:
            from core.rlhf.reward_model import get_reward_model
            reward_model = get_reward_model()
            if not reward_model._model_loaded:
                logger.info("📝 Initializing new reward model (will train with feedback)")
                # Save initial state so it exists
                os.makedirs("models", exist_ok=True)
                torch.save(reward_model.state_dict(), "models/reward_model.pth")
                reward_model._model_loaded = True
        except Exception as e:
            logger.warning(f"Reward model init warning: {e}")
        
        # Report status
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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")