# core/main.py - OPTIMIZED VERSION (< 180s target)
from dotenv import load_dotenv
load_dotenv()
from core.auth.auth_api import router as auth_router
from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from core.intent_router import classify_conversation_intent, is_out_of_domain, get_out_of_domain_message
from pydantic import BaseModel, Field
import asyncio
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os
import httpx
from typing import Optional, Dict, List, Any
import json
from datetime import datetime
import logging
import traceback
import torch
import uuid
import os
from fastapi import HTTPException, Request
import hashlib
import numpy as np
from decimal import Decimal
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
# modules
from core.langgraph import run_multi_agent
from core.analytics import run_causal_analysis, run_shap_analysis
from core.arxiv import retrieve_arxiv_evidence, _get_fallback_papers
from core.utils import cache_set, cache_get
from core.model_loader import get_model_status
from core.utils import load_session_state, save_session_state
from core.mistral import generate_with_mistral
from core.intent_router import classify_conversation_intent
import os
from core.routers import public, dashboard
from core.routers.dashboard import router as dashboard_router
from core.routers.public import router as public_router
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
print(f"CSMODEL_ENABLED: {os.getenv('CSMODEL_ENABLED')}")
print(f"CSMODEL_USE_FROM_PRETRAINED: {os.getenv('CSMODEL_USE_FROM_PRETRAINED')}")
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

# Logging — explicit setup so it works even when uvicorn pre-configures root logger
_log_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)
_stream_handler.setLevel(logging.INFO)

_file_handler = logging.FileHandler("backend_debug.log", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_file_handler.setLevel(logging.INFO)

# Force-configure the root logger (works even if uvicorn already touched it)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in _root_logger.handlers):
    _root_logger.addHandler(_stream_handler)
_root_logger.addHandler(_file_handler)

# Ensure all core sub-loggers propagate up to root (so they appear in console + file)
for _name in ("core.arxiv", "core.langgraph", "core.analytics",
              "core.mistral", "core.model_loader", "core.rlhf", "biomed"):
    _child = logging.getLogger(_name)
    _child.setLevel(logging.INFO)
    _child.propagate = True   # sends records up to root logger → our handlers

logger = logging.getLogger("biomed")

app = FastAPI(title="IXORA - Multi-Agent Research Assistant (Optimized)")

# Mount frontend static files (single mount)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    path = Path("auth/index.html")
    if not path.is_file():
        return "<h1>Error: auth/index.html not found</h1>"
    return path.read_text(encoding="utf-8")

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    path = Path("auth/login.html")
    if not path.is_file():
        return "<h1>Error: auth/login.html not found</h1>"
    return path.read_text(encoding="utf-8")

# Add your /dashboard route directly here too (temporary)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    path = Path("frontend/dashboard.html")   # adjust path if needed
    if not path.is_file():
        return "<h1>Error: dashboard.html not found</h1>"
    return path.read_text(encoding="utf-8")

app.include_router(public_router)     
app.include_router(dashboard_router)

# Config (add to your existing .env variables if missing)
MONGODB_URI      = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME  = os.getenv("MONGODB_DB_NAME", "ixora")
JWT_SECRET       = os.getenv("JWT_SECRET")  # must be set in .env!
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
users = db["users"]
users.create_index("email", unique=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

@app.get("/api/health")
def health():
    try:
        # Try a real command
        db.command("ping")
        # Optional: check if users collection exists
        collections = db.list_collection_names()
        users_exists = "users" in collections
        return {
            "status": "healthy",
            "database": "connected",
            "collections": collections,
            "users_collection_exists": users_exists
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e), "uri_used": MONGODB_URI}


############## FIREBASE ######################

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ─── Firebase Admin SDK initialization ──────────────────────────────────────
try:
    # Build credential dict from environment variables
    cred_dict = {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),  # optional
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),  # optional
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": (
            f"https://www.googleapis.com/robot/v1/metadata/x509/"
            f"{os.getenv('FIREBASE_CLIENT_EMAIL')}"
        ) if os.getenv("FIREBASE_CLIENT_EMAIL") else None,
    }

    # Remove keys with empty values to avoid validation errors
    cred_dict = {k: v for k, v in cred_dict.items() if v is not None and v != ""}

    # Basic required fields check (helps debugging)
    required = ["project_id", "private_key", "client_email"]
    missing = [k for k in required if k not in cred_dict or not cred_dict[k]]
    if missing:
        raise ValueError(f"Missing required Firebase env vars: {', '.join(missing)}")

    cred = credentials.Certificate(cred_dict)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        logger.info(
            f"Firebase Admin SDK initialized successfully | "
            f"project: {cred_dict['project_id']}"
        )
    else:
        logger.info("Firebase Admin SDK already initialized")

except ValueError as ve:
    logger.error(f"Firebase configuration error: {ve}")
    # In development you can continue → in production you may want to raise
except Exception as e:
    logger.exception("Failed to initialize Firebase Admin SDK")
    # Decide: continue degraded or crash startup
    # For now we continue (auth endpoints will 500 until fixed)

# ─── Firebase token verification dependency ─────────────────────────────────
firebase_bearer = HTTPBearer(
    scheme_name="Firebase ID Token",
    auto_error=False  # we'll handle missing token ourselves
)

async def get_current_firebase_user(
    auth_header: HTTPAuthorizationCredentials | None = Depends(firebase_bearer)
) -> dict:
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": 'Bearer error="no_token"'},
        )

    token = auth_header.credentials

    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        # check_revoked=True adds extra security (requires network call)
        # remove it in very high-traffic scenarios if latency becomes issue
        return decoded_token
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Firebase ID token has expired")
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Firebase ID token has been revoked")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")
    except auth.CertificateFetchError:
        raise HTTPException(
            status_code=503,
            detail="Firebase certificate fetch failed (network/cert issue)"
        )
    except Exception as e:
        logger.error(f"Unexpected Firebase verify error: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


# ─── Firebase-powered Google login endpoint ─────────────────────────────────
@app.post("/api/auth/google")
async def firebase_google_login(
    firebase_user: dict = Depends(get_current_firebase_user)
):
    email = firebase_user.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email not provided by Firebase")

    email = email.lower().strip()

    existing = users.find_one({"email": email})

    if existing:
        update_fields = {
            "last_login": datetime.utcnow(),
        }
        if picture := firebase_user.get("picture"):
            update_fields["google_picture"] = picture

        users.update_one(
            {"_id": existing["_id"]},
            {"$set": update_fields}
        )
        user_doc = users.find_one({"_id": existing["_id"]})
    else:
        # Split name safely
        full_name = firebase_user.get("name", "").strip()
        name_parts = full_name.split(" ", 1) if full_name else ["", ""]
        first_name = name_parts[0].strip()
        last_name = name_parts[1].strip() if len(name_parts) > 1 else ""

        user_id = str(uuid.uuid4())
        user_doc = {
            "_id": user_id,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "provider": "google",
            "firebase_uid": firebase_user["uid"],
            "google_picture": firebase_user.get("picture", ""),
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow(),
            # Add defaults you had in auth_api.py register
            "preferences": {"domain": "biomed", "theme": "light"},
            "is_verified": firebase_user.get("email_verified", False),
        }
        users.insert_one(user_doc)

    # Issue your own JWT (same as before)
    token = create_token(str(user_doc["_id"]), email)

    return {
        "token": token,
        "user": {
            "id": str(user_doc["_id"]),
            "email": email,
            "first_name": user_doc.get("first_name", ""),
            "last_name": user_doc.get("last_name", ""),
            "provider": user_doc.get("provider", "google"),
            "picture": user_doc.get("google_picture", ""),
        },
        "message": f"Welcome, {user_doc.get('first_name') or email.split('@')[0]}!"
    }

##########################

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=24*7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = users.find_one({"_id": payload["sub"]})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ========== AUTH ROUTES ==========

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    code: str

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    user = users.find_one({"email": req.email.lower().strip()})
    if not user or not pwd_context.verify(req.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail={"message": "Invalid email or password"})
    token = create_token(str(user["_id"]), user["email"])
    return {
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "first_name": user.get("first_name", ""),
            "last_name": user.get("last_name", ""),
            "provider": user.get("provider", "email"),
        }
    }

@app.post("/api/auth/register")
async def auth_register(req: RegisterRequest):
    if users.find_one({"email": req.email.lower().strip()}):
        raise HTTPException(status_code=400, detail={"message": "Email already registered"})
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail={"message": "Password must be at least 8 characters"})
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "email": req.email.lower().strip(),
        "first_name": req.first_name.strip(),
        "last_name": req.last_name.strip(),
        "password_hash": pwd_context.hash(req.password),
        "provider": "email",
        "created_at": datetime.utcnow().isoformat(),
    }
    users.insert_one(user_doc)
    token = create_token(user_id, user_doc["email"])
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": user_doc["email"],
            "first_name": user_doc["first_name"],
            "last_name": user_doc["last_name"],
            "provider": "email",
        }
    }

@app.post("/api/auth/google")
async def auth_google(req: GoogleAuthRequest):
    async with httpx.AsyncClient() as client_http:
        # Exchange code for tokens
        token_res = await client_http.post("https://oauth2.googleapis.com/token", data={
            "code": req.code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": "http://localhost:3000/auth/callback",
            "grant_type": "authorization_code",
        })
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail={"message": "Google token exchange failed"})
        tokens = token_res.json()
        # Get user info
        info_res = await client_http.get("https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"})
        if info_res.status_code != 200:
            raise HTTPException(status_code=400, detail={"message": "Could not fetch Google user info"})
        guser = info_res.json()
        email = guser.get("email", "").lower().strip()
        existing = users.find_one({"email": email})
        if existing:
            user_id = str(existing["_id"])
        else:
            user_id = str(uuid.uuid4())
            users.insert_one({
                "_id": user_id,
                "email": email,
                "first_name": guser.get("given_name", ""),
                "last_name": guser.get("family_name", ""),
                "password_hash": "",
                "provider": "google",
                "created_at": datetime.utcnow().isoformat(),
            })
            existing = users.find_one({"_id": user_id})
        token = create_token(user_id, email)
        return {
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "first_name": existing.get("first_name", ""),
                "last_name": existing.get("last_name", ""),
                "provider": "google",
            }
        }


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

# In main.py - Update run_optimization_background

async def run_optimization_background(session_id: str, parameters: Dict[str, Any], domain: str):
    """Background task that runs Bayesian optimization"""
    logger.info(f"🔄 [BACKGROUND] Starting Bayesian optimization for session {session_id}")
    start_time = time.time()
    
    try:
        # FIX: Ensure parameters are in the right format
        opt_parameters = {}
        for param_name, param_value in parameters.items():
            if isinstance(param_value, dict):
                # Extract value from parameter dict
                opt_parameters[param_name] = param_value.get("value", param_value)
            else:
                opt_parameters[param_name] = param_value
        
        # Run optimization with timeout
        opt_result = await asyncio.wait_for(
            run_bayesian_optimization(opt_parameters, domain=domain),
            timeout=30.0
        )
        
        # FIX: Safely extract optimal_parameters
        optimal_params = opt_result.get("optimal_parameters", {})
        if not optimal_params and "best_parameters" in opt_result:
            optimal_params = opt_result.get("best_parameters", {})
        
        # Load and update session state
        session_state = load_session_state(session_id) or {}
        
        session_state["bayesian_optimization"] = {
            "status": opt_result.get("status", "completed"),
            "result": opt_result,
            "optimal_parameters": optimal_params,  # Explicitly store
            "duration": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "parameters_analyzed": list(parameters.keys())
        }
        
        save_session_state(session_id, session_state)
        
        logger.info(f"✅ [BACKGROUND] Optimization completed in {time.time() - start_time:.2f}s")
        logger.info(f"   Result: {opt_result.get('status', 'unknown')}")
        
    except asyncio.TimeoutError:
        logger.error("❌ [BACKGROUND] Optimization timed out after 30s")
        # Store timeout status
        session_state = load_session_state(session_id) or {}
        session_state["bayesian_optimization"] = {
            "status": "timeout",
            "error": "Optimization timed out after 30 seconds",
            "timestamp": datetime.now().isoformat()
        }
        save_session_state(session_id, session_state)
    
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

#+=======================

def log_pipeline_progress(step: str, duration: float, details: dict = None):
    """Log pipeline progress with timing"""
    logger.info(f"🔄 [{step.upper()}] Completed in {duration:.2f}s")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")

# ========== PIPELINE LOGGING HELPERS ==========

def log_pipeline_progress(step: str, duration: float, details: dict = None):
    """Log pipeline progress with timing"""
    logger.info(f"🔄 [{step.upper()}] Completed in {duration:.2f}s")
    if details:
        for key, value in details.items():
            logger.info(f"   {key}: {value}")

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

# ========== MAIN CHAT ENDPOINT (OPTIMIZED) ==========

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id or str(uuid.uuid4())
    forced_domain = (req.domain or "biomed").lower()

    logger.info(f"🎯 FORCED DOMAIN: {forced_domain.upper()} | Session: {session_id}")
    logger.info(f"📝 Query: {req.message[:100]}...")

    start_time = time.time()

    # Load or initialize session
    session_state = load_session_state(session_id) or {}
    history = session_state.get("history", [])
    history.append({"role": "user", "content": req.message})

    try:
        # ── 1. Intent classification ────────────────────────────────────────
        intent_start = time.time()
        intent_info = await classify_conversation_intent(
            query=req.message,
            session_state=session_state,           # gives context if needed
            forced_domain=forced_domain
        )
        intent_time = time.time() - intent_start

        intent = intent_info["intent"]
        needs_pipeline = intent_info.get("needs_pipeline", False)
        confidence_from_classifier = intent_info.get("confidence", 0.7)

        logger.info(
            f"🤖 Intent: {intent} | needs_pipeline={needs_pipeline} | "
            f"conf={confidence_from_classifier:.2f} ({intent_time:.2f}s)"
        )

        # ── 2. Strict domain boundary enforcement ───────────────────────────
        if is_out_of_domain(intent):
            refusal_message = get_out_of_domain_message(intent, req.message)
            
            logger.warning(f"❌ OUT-OF-DOMAIN: {intent}")
            
            history.append({"role": "assistant", "content": refusal_message})
            session_state.update({
                "history": history[-20:],
                "last_query": req.message,
                "last_response": refusal_message,
                "intent": intent,
                "domain_violation": {
                    "detected": intent,
                    "query": req.message,
                    "timestamp": datetime.now().isoformat()
                },
                "domain": forced_domain,
                "timestamp": datetime.now().isoformat()
            })
            save_session_state(session_id, session_state)

            total_time = time.time() - start_time
            return {
                "response": refusal_message,
                "session_id": session_id,
                "confidence": 1.0,
                "trace": [{
                    "step": "domain_check",
                    "result": "rejected",
                    "reason": f"Query classified as {intent}",
                    "duration": round(total_time, 2)
                }],
                "parameters": {},
                "processing_time_seconds": round(total_time, 2),
                "intent": intent,
                "domain": forced_domain,
                "used_full_pipeline": False,
                "domain_violation": True
            }

        # ── 3. Decide response path ─────────────────────────────────────────
        response_text = ""
        trace = []
        parameters = {}
        white_box = {}
        optimization_launched = False
        used_full_pipeline = False
        reward_score = None  # Initialize reward score

        if not needs_pipeline:
            # ── Fast paths: casual chat, explanations, clarifications ───────
            if intent in ["casual_chat", "clarification", "meta_off_topic"]:
                system_content = f"""You are an expert assistant in {forced_domain.upper()} topics.
                    Be precise, helpful, and stay within {forced_domain} unless explicitly asked otherwise.
                    Use clear, structured answers when explaining concepts."""
                history_text = "\n\n".join(
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in history[:-1]
                )
                prompt = f"{history_text}\n\nUSER: {req.message}\n\nASSISTANT:"
                
                response_text, _ = await generate_with_mistral(
                    prompt=prompt,
                    system_prompt=system_content,
                    max_tokens=900,
                    temperature=0.75,
                    domain=forced_domain
                )

            elif intent in ["explanation", "request_explanation"]:
                explain_system = (
                    "You are an expert explainer. Provide clear, detailed, "
                    "well-structured explanations. Use context when relevant."
                )
                history_text = "\n\n".join(
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in history[:-1]
                )
                prompt = (
                    f"{history_text}\n\n"
                    f"USER: Explain in detail: {req.message}\n\n"
                    "ASSISTANT:"
                )
                
                response_text, _ = await generate_with_mistral(
                    prompt=prompt,
                    system_prompt=explain_system,
                    max_tokens=1400,
                    temperature=0.7,
                    explanation_mode=True,
                    domain=forced_domain
                )

            else:
                # Fallback short answer for any other light intent
                response_text = await generate_with_mistral(req.message, intent_info)
            
            # Get RLHF reward score for fast-path responses
            try:
                from core.rlhf.reward_model import get_reward_model
                reward_model = get_reward_model()
                
                if response_text and len(response_text) > 10:
                    with torch.no_grad():
                        reward_tensor = reward_model([response_text])
                        reward_score = float(reward_tensor.item())
                        logger.info(f"📊 RLHF Reward (fast path): {reward_score:.4f}")
                        
            except Exception as e:
                logger.warning(f"Could not compute RLHF reward for fast path: {e}")
                reward_score = None

        else:
            # ── Full multi-agent research pipeline ──────────────────────────
            logger.info(f"🔬 Starting full {forced_domain.upper()} pipeline...")
            pipeline_start = time.time()

            result = await run_multi_agent(
                query=req.message,
                domain=forced_domain,
                session_id=session_id,
                history=history
            )

            pipeline_time = time.time() - pipeline_start
            logger.info(f"Pipeline completed in {pipeline_time:.2f}s")

            response_text = result.get("final_response", "Generation failed.")
            trace = result.get("trace", [])
            confidence_from_classifier = result.get("confidence", confidence_from_classifier)
            white_box = result.get("white_box_state", {})
            raw_parameters = white_box.get("parameters", {})
            
            # Get reward score from pipeline result
            reward_score = result.get("reward_score", None)
            if reward_score is None:
                # Fallback: compute reward score if not provided by pipeline
                try:
                    from core.rlhf.reward_model import get_reward_model
                    reward_model = get_reward_model()
                    
                    if response_text and len(response_text) > 10:
                        with torch.no_grad():
                            reward_tensor = reward_model([response_text])
                            reward_score = float(reward_tensor.item())
                            logger.info(f"📊 RLHF Reward (fallback): {reward_score:.4f}")
                            
                except Exception as e:
                    logger.warning(f"Could not compute RLHF reward (fallback): {e}")
                    reward_score = None

            # Normalize parameters
            parameters = {}
            for name, val in raw_parameters.items():
                clean_name = name.strip().replace(" ", "_").lower()
                if isinstance(val, dict):
                    parameters[clean_name] = {
                        "value": val.get("value", ""),
                        "unit": val.get("unit", ""),
                        "confidence": val.get("confidence", 0.8),
                        "raw_text": val.get("raw_text", str(val)),
                        "method": val.get("method", "extracted")
                    }
                else:
                    parameters[clean_name] = {
                        "value": val,
                        "unit": "",
                        "confidence": 0.8,
                        "raw_text": str(val),
                        "method": "auto"
                    }

            used_full_pipeline = True

            # Launch background Bayesian optimization if appropriate
            if parameters:
                has_optimizable = any(
                    isinstance(p.get("value"), (int, float)) or
                    (isinstance(p.get("value"), list) and len(p["value"]) == 2 and
                     all(isinstance(x, (int, float)) for x in p["value"]))
                    for p in parameters.values()
                )
                if has_optimizable:
                    opt_params = {k: v["value"] if isinstance(v, dict) else v for k, v in parameters.items()}
                    background_tasks.add_task(
                        run_optimization_background,
                        session_id,
                        opt_params,
                        forced_domain
                    )
                    optimization_launched = True

        # ── 4. Update session history (common path) ────────────────────────
        history.append({"role": "assistant", "content": response_text})
        # Record the very first research query so arXiv results stay pinned to it
        # across all follow-up turns in this session.
        if used_full_pipeline and "first_research_query" not in session_state:
            session_state["first_research_query"] = req.message

        session_state.update({
            "history": history[-20:],
            "last_query": req.message,
            "last_response": response_text,
            "parameters": parameters,
            "trace": trace,
            "confidence": confidence_from_classifier,
            "domain": forced_domain,
            "intent": intent,
            "used_full_pipeline": used_full_pipeline,
            "optimization_launched": optimization_launched,
            "reward_score": reward_score,  # Store reward score in session
            "timestamp": datetime.now().isoformat()
        })

        if white_box and "analytics" in white_box:
            session_state["analytics"] = white_box["analytics"]

        save_session_state(session_id, session_state)

        # ── 5. Build final response with RLHF reward ────────────────────────
        total_time = time.time() - start_time
        
        # Log successful response for RLHF monitoring
        if reward_score is not None:
            logger.info(f"🎯 Response scored: {reward_score:.4f} (higher is better)")
        
        response_data = {
            "response": response_text,
            "session_id": session_id,
            "confidence": round(confidence_from_classifier, 3),
            "reward_score": reward_score,  # Include reward score in response
            "trace": trace[:10] if trace else [],
            "parameters": parameters,
            "processing_time_seconds": round(total_time, 2),
            "intent": intent,
            "domain": forced_domain,
            "used_full_pipeline": used_full_pipeline,
            "optimization_note": "Running in background" if optimization_launched else None
        }
        
        # Log for RLHF monitoring (optional)
        if reward_score is not None and used_full_pipeline:
            try:
                # Hash the query for feedback tracking
                import hashlib
                query_hash = hashlib.sha256(req.message.encode()).hexdigest()[:16]
                
                # Log to RLHF monitoring
                rlhf_log = {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "query_hash": query_hash,
                    "query": req.message[:200],
                    "reward_score": reward_score,
                    "response_length": len(response_text),
                    "pipeline_used": used_full_pipeline,
                    "intent": intent
                }
                
                # Save to RLHF monitoring log
                rlhf_log_file = "logs/rlhf_monitoring.jsonl"
                os.makedirs("logs", exist_ok=True)
                with open(rlhf_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rlhf_log, ensure_ascii=False) + "\n")
                    
                logger.debug(f"📊 RLHF monitoring logged: {reward_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to log RLHF monitoring: {e}")

        return response_data

    except Exception as e:
        logger.exception(f"Chat endpoint failed: {e}")
        return {
            "error": str(e),
            "session_id": session_id,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
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
                    retrieve_arxiv_evidence(req.query, max_papers=80),
                    timeout=60.0
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
        
        # Build response - only include error if it exists
        response = {
            "status": opt_data.get("status", "unknown"),
            "result": opt_data.get("result", {}),
            "timestamp": opt_data.get("timestamp", "")
        }
        
        # Only include error field if there's actually an error
        if "error" in opt_data and opt_data["error"]:
            response["error"] = opt_data["error"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve optimization status")


# ========== ARXIV ENDPOINT ==========

@app.post("/arxiv")
async def arxiv_search(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(422, detail="Invalid or missing JSON body")

    query = payload.get("query")
    if not query or not isinstance(query, str) or len(query.strip()) < 3:
        raise HTTPException(
            422,
            detail="Missing or invalid 'query' field. Send something like: {'query': 'yeast biomass pH'}"
        )

    session_id = payload.get("session_id")  # frontend should always pass this

    # User-controlled paper count: default 20, capped at 100
    requested = payload.get("max_papers", 20)
    try:
        max_papers = max(1, min(int(requested), 100))
    except (TypeError, ValueError):
        max_papers = 20

    # ── Session-aware query pinning ──────────────────────────────────────────
    # On follow-up turns the frontend sends the follow-up text as , but
    # the arXiv papers should always reflect the ORIGINAL research question so
    # they stay relevant.  We use two layers of stickiness:
    #   1. If papers were already cached for this session → return them as-is.
    #   2. Otherwise resolve the canonical research query from session state
    #      (first_research_query) so the search is anchored to the topic, not
    #      the follow-up wording.
    if session_id:
        session_state = load_session_state(session_id)

        # Layer 1: return cached papers immediately (no new fetch needed)
        cached_papers = session_state.get("arxiv_papers")
        if cached_papers:
            logger.info(
                f"📚 Returning {len(cached_papers)} cached arXiv papers for session {session_id}"
            )
            return {
                "links": cached_papers,
                "count": len(cached_papers),
                "status": "success",
                "source": "session_cache",
            }

        # Layer 2: pin query to the first research question in this session
        first_research_query = session_state.get("first_research_query")
        if first_research_query:
            logger.info(
                f"📌 Pinning arXiv search to original query: '{first_research_query[:80]}...'"
            )
            query = first_research_query

    logger.info(f"📄 User requested {max_papers} papers for query: '{query.strip()}'")

    try:
        # Try arXiv API — timeout scales with requested paper count
        try:
            papers = await asyncio.wait_for(
                retrieve_arxiv_evidence(query, max_papers=max_papers),
                timeout=max(30, max_papers * 0.75)  # ~30s min, scales up for larger requests
            )

            if papers:
                logger.info(f"📚 Found {len(papers)} arXiv papers")
                # Cache papers in session so follow-up calls return the same set
                if session_id:
                    session_state = load_session_state(session_id) or {}
                    session_state["arxiv_papers"] = papers
                    session_state.setdefault("first_research_query", query.strip())
                    save_session_state(session_id, session_state)
                return {
                    "links": papers,
                    "count": len(papers),
                    "status": "success",
                    "source": "arxiv_api",
                }

        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"arXiv API failed: {e}, using fallback")

        query = query.strip()

        # Use fallback papers
        fallback_papers = _get_fallback_papers(query)
        logger.info(f"📚 Using {len(fallback_papers)} fallback papers")

        # Cache fallback too so follow-ups are stable
        if session_id:
            session_state = load_session_state(session_id) or {}
            session_state["arxiv_papers"] = fallback_papers
            session_state.setdefault("first_research_query", query)
            save_session_state(session_id, session_state)

        return {
            "links": fallback_papers,
            "count": len(fallback_papers),
            "status": "partial",
            "source": "fallback",
            "note": "arXiv API unavailable, showing relevant reference papers",
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
async def receive_feedback(request: Request):
    """Receive user feedback (thumbs up/down) for RLHF training"""
    try:
        data = await request.json()
    except:
        raise HTTPException(422, detail="Invalid or missing JSON body")
    
    session_id = data.get("session_id", "")
    preference = data.get("preference", "").lower()
    response_text = data.get("response", "")
    query_hash = data.get("query_hash", "unknown")
    reason = data.get("reason", "")
    
    if preference not in ["good", "bad"]:
        raise HTTPException(400, detail="Preference must be 'good' or 'bad'")
    
    if not session_id:
        raise HTTPException(400, detail="Missing session_id")
    
    try:
        # Get query text from session if available
        query_text = ""
        session_state = load_session_state(session_id)
        if session_state:
            query_text = session_state.get("last_query", "")
        
        # Log feedback with context
        from core.rlhf.feedback_logger import log_feedback_with_context
        success = log_feedback_with_context(
            session_id=session_id,
            preference=preference,
            response_text=response_text,
            query_hash=query_hash,
            query_text=query_text,
            reason=reason
        )
        
        if success:
            logger.info(f"👍 Feedback recorded: {preference} for session {session_id[:8]}")
            return {
                "status": "success", 
                "message": f"Feedback recorded ({preference})",
                "feedback_count": _count_feedbacks()  # Optional: return count
            }
        else:
            raise HTTPException(500, detail="Failed to save feedback")
            
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}", exc_info=True)
        raise HTTPException(500, detail="Failed to process feedback")

def _count_feedbacks():
    """Helper to count feedback entries"""
    try:
        feedback_file = "logs/rlhf_feedback.jsonl"
        if os.path.exists(feedback_file):
            with open(feedback_file, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
    except:
        pass
    return 0

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
    logger.info("🚀 IXORA - Starting up (OPTIMIZED VERSION with RLHF)")
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
        
        # Initialize RLHF system (in background)
        logger.info("🧠 Initializing RLHF system...")
        try:
            from core.rlhf import initialize_rlhf_system, check_rlhf_status
            
            # Initialize RLHF in background thread (non-blocking)
            import threading
            
            def initialize_rlhf_background():
                try:
                    success = initialize_rlhf_system(warmup=True)
                    if success:
                        status = check_rlhf_status()
                        logger.info(f"✅ RLHF initialized: Model trained={status.get('model_trained', False)}, "
                                   f"Feedbacks={status.get('feedback_count', 0)}")
                    else:
                        logger.warning("⚠️ RLHF initialization had issues")
                except Exception as e:
                    logger.error(f"❌ RLHF background initialization failed: {e}")
            
            rlhf_thread = threading.Thread(
                target=initialize_rlhf_background,
                daemon=True
            )
            rlhf_thread.start()
            
            # Give it a moment to start, then check initial status
            await asyncio.sleep(1)
            try:
                rlhf_status = check_rlhf_status()
                logger.info(f"📊 RLHF Status: Model trained={rlhf_status.get('model_trained', False)}, "
                           f"Feedbacks={rlhf_status.get('feedback_count', 0)}")
                
                if rlhf_status.get("feedback_count", 0) >= 10 and not rlhf_status.get("model_trained", False):
                    logger.info("📈 RLHF: Enough feedback collected, ready for training")
                    
                    # Optionally start training in background if enough feedback
                    if os.getenv("RLHF_AUTO_TRAIN", "true").lower() == "true":
                        def train_rlhf_background():
                            try:
                                from core.rlhf import train_on_existing_feedback
                                logger.info("🔄 Starting RLHF training in background...")
                                success = train_on_existing_feedback()
                                if success:
                                    logger.info("🎉 RLHF training completed!")
                                else:
                                    logger.warning("⚠️ RLHF training completed with issues")
                            except Exception as e:
                                logger.error(f"RLHF background training failed: {e}")
                        
                        train_thread = threading.Thread(target=train_rlhf_background, daemon=True)
                        train_thread.start()
                        
            except Exception as e:
                logger.warning(f"Could not check RLHF status: {e}")
                
        except ImportError as e:
            logger.warning(f"RLHF module not available: {e}. Continuing without RLHF.")
        except Exception as e:
            logger.error(f"Failed to initialize RLHF system: {e}. Continuing without RLHF.")
        
        # Initialize reward model (fallback if RLHF module fails)
        try:
            from core.rlhf.reward_model import get_reward_model, initialize_reward_model
            logger.info("📝 Initializing reward model...")
            
            # Create directories
            os.makedirs("models", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Initialize model
            initialize_reward_model()
            reward_model = get_reward_model()
            
            # Test the model
            if not is_dev:  # Only test in production for speed
                with torch.no_grad():
                    test_text = "This is a test response."
                    score = reward_model([test_text])
                    logger.info(f"✅ Reward model test passed: {score.item():.4f}")
            
        except Exception as e:
            logger.warning(f"Reward model initialization warning: {e}")
        
        # Get and log model status
        try:
            status = get_model_status()
            loaded = sum(1 for s in status.values() if s["loaded"])
            total = len(status)
            
            logger.info(f"📊 Models: {loaded}/{total} loaded successfully")
            for name, info in status.items():
                symbol = "✅" if info["loaded"] else "🔄" if info["loading"] else "❌"
                logger.info(f"   {symbol} {name}: {'Ready' if info['loaded'] else 'Loading' if info['loading'] else 'Failed'}")
            
            # Check BioMistral responsiveness
            biomistral_status = status.get("biomistral", {})
            if biomistral_status.get("loaded"):
                try:
                    from core.model_loader import model_loader
                    test_result = await model_loader.generate_with_biomistral("Test", max_tokens=5)
                    logger.info(f"✅ BioMistral responsive: {test_result[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️ BioMistral loaded but not responsive: {e}")
        
        except Exception as e:
            logger.warning(f"Could not get model status: {e}")
        
        # Check for existing feedback to log
        try:
            feedback_file = "logs/rlhf_feedback.jsonl"
            if os.path.exists(feedback_file):
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedback_count = sum(1 for line in f if line.strip())
                logger.info(f"📊 Found {feedback_count} existing RLHF feedback entries")
                
                # If enough feedback but RLHF not trained, log warning
                if feedback_count >= 10:
                    try:
                        from core.rlhf.reward_model import get_reward_model
                        model = get_reward_model()
                        if hasattr(model, 'is_trained') and not model.is_trained():
                            logger.warning(f"⚠️ {feedback_count} feedbacks available but model not trained yet")
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Could not check feedback count: {e}")
        
        logger.info("✅ System ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        logger.warning("⚠️ Continuing with lazy loading mode")
        
        # Emergency initialization of critical components
        try:
            # Ensure directories exist
            os.makedirs("models", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Initialize basic reward model
            try:
                from core.rlhf.reward_model import get_reward_model
                model = get_reward_model()
                logger.info("✅ Basic reward model initialized")
            except:
                logger.warning("❌ Could not initialize reward model")
            
            # Create empty feedback file if needed
            feedback_file = "logs/rlhf_feedback.jsonl"
            if not os.path.exists(feedback_file):
                with open(feedback_file, "w", encoding="utf-8") as f:
                    pass
                logger.info("✅ Created empty feedback log")
                
        except Exception as init_error:
            logger.error(f"Emergency initialization failed: {init_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        log_config=None,   # don't let uvicorn overwrite our logging setup
    )