"""
IXORA Authentication API
FastAPI backend for user registration, login, and Google OAuth.
Run: uvicorn auth_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os
import httpx
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────
MONGODB_URI      = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME  = os.getenv("MONGODB_DB_NAME", "ixora")
JWT_SECRET       = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = 24 * 7   # 7 days

"""GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3000/auth/callback")"""

# ─── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="IXORA Auth API", version="1.0.0")

# Replace or update your existing middleware block with this:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # your React dev server
        "http://127.0.0.1:3000",       # sometimes browsers use 127.0.0.1
        "http://localhost:5173",       # if using Vite default port sometimes
        # add "https://your-production-domain.com" later
    ],
    allow_credentials=True,            # needed if you ever send cookies/auth headers
    allow_methods=["*"],               # GET, POST, OPTIONS, etc.
    allow_headers=["*"],               # Content-Type, Authorization, etc.
)

# ─── DB & Security ──────────────────────────────────────────────────────────
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]
users_col = db["users"]
users_col.create_index("email", unique=True)

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()


# ─── Schemas ────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

    @validator("password")
    def pw_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @validator("first_name", "last_name")
    def name_nonempty(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleCallbackRequest(BaseModel):
    code: str


class TokenResponse(BaseModel):
    token: str
    user: dict
    message: str


# ─── Helpers ────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def sanitize_user(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "first_name": user.get("first_name", ""),
        "last_name": user.get("last_name", ""),
        "provider": user.get("provider", "email"),
        "created_at": user.get("created_at", "").isoformat() if user.get("created_at") else "",
    }

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    payload = decode_token(credentials.credentials)
    user = users_col.find_one({"email": payload["email"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "IXORA Auth API", "status": "running"}


@app.post("/api/auth/register", response_model=TokenResponse)
async def register(body: RegisterRequest):
    # Check existing
    if users_col.find_one({"email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "An account with this email already exists."}
        )

    user_doc = {
        "first_name": body.first_name,
        "last_name": body.last_name,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "provider": "email",
        "is_verified": False,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "preferences": {"domain": "biomed", "theme": "light"},
    }

    result = users_col.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    token = create_token(str(result.inserted_id), body.email)

    return TokenResponse(
        token=token,
        user=sanitize_user(user_doc),
        message=f"Welcome to IXORA, {body.first_name}!"
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = users_col.find_one({"email": body.email})

    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid email or password."}
        )

    # Update last login
    users_col.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})

    token = create_token(str(user["_id"]), user["email"])
    return TokenResponse(
        token=token,
        user=sanitize_user(user),
        message=f"Welcome back, {user.get('first_name', '')}!"
    )


"""@app.post("/api/auth/google", response_model=TokenResponse)
async def google_oauth(body: GoogleCallbackRequest):
    #Exchange Google auth code for user info and return JWT.
    async with httpx.AsyncClient() as client_http:
        # Exchange code for tokens
        token_res = await client_http.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": body.code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            }
        )
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail={"message": "Google OAuth failed."})

        tokens = token_res.json()
        access_token = tokens.get("access_token")

        # Get user info
        userinfo_res = await client_http.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        if userinfo_res.status_code != 200:
            raise HTTPException(status_code=400, detail={"message": "Failed to retrieve Google profile."})

        guser = userinfo_res.json()
        email = guser.get("email")
        first_name = guser.get("given_name", "")
        last_name = guser.get("family_name", "")
        google_id = guser.get("id")
        picture = guser.get("picture", "")

    # Upsert user
    existing = users_col.find_one({"email": email})
    if existing:
        users_col.update_one(
            {"_id": existing["_id"]},
            {"$set": {"last_login": datetime.utcnow(), "google_picture": picture}}
        )
        user_doc = users_col.find_one({"email": email})
    else:
        user_doc = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "google_id": google_id,
            "google_picture": picture,
            "provider": "google",
            "is_verified": True,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "preferences": {"domain": "biomed", "theme": "light"},
        }
        result = users_col.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id

    token = create_token(str(user_doc["_id"]), email)
    return TokenResponse(
        token=token,
        user=sanitize_user(user_doc),
        message=f"Welcome, {first_name}!"
    )"""


@app.post("/api/auth/forgot-password")
async def forgot_password(body: dict):
    email = body.get("email", "")
    user = users_col.find_one({"email": email})
    # Always return success to prevent email enumeration
    return {"message": "If that email exists, a reset link has been sent."}


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {"user": sanitize_user(current_user)}


@app.post("/api/auth/logout")
def logout():
    # JWT is stateless; client should discard token
    return {"message": "Logged out successfully."}


@app.get("/api/health")
def health():
    try:
        db.command("ping")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": str(e)}

# Add this to auth_api.py
router = app.router
