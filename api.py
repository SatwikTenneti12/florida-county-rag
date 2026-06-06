import os
import sys
import json
import jwt
import logging
import requests
import re
from datetime import datetime, timedelta
from typing import Optional, List, Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.answer_engine import get_answer_engine
from src.rag.retriever import get_retriever
from src.utils.county_normalizer import get_canonical_counties
from src.config import TOPICS, CHUNKS_PATH
from src.auth import database as db
from src.classification.guardrails import check_input_safety
from src.utils.emailer import send_verification_email, validate_email_settings, EmailConfigurationError

# JWT Secret config
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY", "").strip()
DISABLE_RECAPTCHA = os.getenv("DISABLE_RECAPTCHA", "").strip().lower() in {"1", "true", "yes"}

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

app = FastAPI(title="Florida County RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chunk_cache = None


def load_chunk_cache():
    global _chunk_cache
    if _chunk_cache is None:
        chunks = []
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        _chunk_cache = chunks
    return _chunk_cache


def lexical_search_chunks(query: str, county: Optional[str], top_k: int):
    terms = [t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 2]
    phrase = query.lower().strip()
    scored = []

    for chunk in load_chunk_cache():
        if county and chunk.get("county") != county:
            continue

        text = chunk.get("text", "")
        text_lower = text.lower()
        score = 0
        if phrase and phrase in text_lower:
            score += 10
        for term in terms:
            score += min(text_lower.count(term), 5)

        if score <= 0:
            continue

        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, chunk in scored[:top_k]:
        results.append({
            "text": chunk.get("text", ""),
            "county": chunk.get("county", "Unknown"),
            "pdf_file": chunk.get("pdf_file", "Unknown"),
            "page_start": int(chunk.get("page_start", 0)),
            "page_end": int(chunk.get("page_end", 0)),
            "distance": round(1 / (1 + score), 4),
            "citation": f"{chunk.get('county', 'Unknown')}, {chunk.get('pdf_file', 'Unknown')}, pp. {chunk.get('page_start', 0)}-{chunk.get('page_end', 0)}",
            "retrieval_mode": "lexical_fallback",
        })
    return results

# ---------------- Auth Logic ----------------

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    county: Optional[str] = None
    captcha_token: str

class LoginRequest(BaseModel):
    email: str
    password: str
    captcha_token: str

def verify_captcha_token(captcha_token: str, remote_ip: Optional[str] = None) -> None:
    if DISABLE_RECAPTCHA:
        logger.warning("CAPTCHA verification is disabled by DISABLE_RECAPTCHA.")
        return

    if not captcha_token:
        raise HTTPException(status_code=400, detail="Missing CAPTCHA")
    if not RECAPTCHA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CAPTCHA verification is not configured.")

    payload = {
        "secret": RECAPTCHA_SECRET_KEY,
        "response": captcha_token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip

    try:
        resp = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("CAPTCHA verification request failed: %s", e)
        raise HTTPException(status_code=503, detail="CAPTCHA verification is temporarily unavailable.")

    if not data.get("success"):
        logger.warning("CAPTCHA verification failed: %s", data.get("error-codes", []))
        raise HTTPException(status_code=400, detail="CAPTCHA verification failed.")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.get_user_by_email(user_email)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# ---------------- Endpoints ----------------

import random

class VerifyRequest(BaseModel):
    email: str
    code: str

@app.post("/api/signup")
async def signup(request: SignupRequest, fastapi_request: Request):
    verify_captcha_token(request.captcha_token, fastapi_request.client.host if fastapi_request.client else None)
    try:
        validate_email_settings()
    except EmailConfigurationError as e:
        logger.error("Email configuration error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))

    success = db.create_user(request.name, request.email, request.county, request.password, otp)
    if not success:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    user = db.get_user_by_email(request.email)
    db.log_activity(user["id"], "signup")
    
    try:
        send_verification_email(request.email, otp)
    except EmailConfigurationError as e:
        logger.error("Email configuration error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Failed to send verification email: %s", e)
        raise HTTPException(status_code=503, detail="Failed to send verification email.")
    
    return {"message": "Verification code sent. Please check your email.", "email": request.email}

@app.post("/api/verify")
async def verify(request: VerifyRequest):
    success = db.verify_user(request.email, request.code)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid verification code or email.")
        
    user = db.get_user_by_email(request.email)
    db.log_activity(user["id"], "verified_email")
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": user["email"], "name": user["name"]}}

class ResendRequest(BaseModel):
    email: str

@app.post("/api/resend_code")
async def resend_code(request: ResendRequest):
    user = db.get_user_by_email(request.email)
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")
    if user["is_verified"] == 1:
        raise HTTPException(status_code=400, detail="Account is already verified.")
        
    # Generate new 6-digit OTP
    otp = str(random.randint(100000, 999999))
    success = db.update_verification_code(request.email, otp)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update verification code.")
        
    db.log_activity(user["id"], "resend_code")
    
    try:
        send_verification_email(request.email, otp)
    except EmailConfigurationError as e:
        logger.error("Email configuration error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Failed to resend verification email: %s", e)
        raise HTTPException(status_code=503, detail="Failed to resend verification email.")
    
    return {"message": "Verification code resent successfully."}

@app.post("/api/login")
async def login(request: LoginRequest, fastapi_request: Request):
    verify_captcha_token(request.captcha_token, fastapi_request.client.host if fastapi_request.client else None)
        
    user = db.get_user_by_email(request.email)
    if not user:
        raise HTTPException(status_code=404, detail="You need to create an account first.")
        
    if user["is_verified"] == 0:
        raise HTTPException(status_code=403, detail="Please verify your email before logging in.")
    
    if not db.verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
        
    db.log_activity(user["id"], "login")
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": user["email"], "name": user["name"]}}

class AskRequest(BaseModel):
    question: str
    county: Optional[str] = None
    top_k: int = 8

class SearchRequest(BaseModel):
    query: str
    county: Optional[str] = None
    top_k: int = 8

class TopicAnalyzeRequest(BaseModel):
    county: str
    top_k: int = 3

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    sources: List[dict] = []

@app.post("/api/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, payload: AskRequest, current_user: dict = Depends(get_current_user)):
    # 1. Guardrail Security Check
    status, justification = await check_input_safety(payload.question)
    
    if status == "BLOCK":
        # Log the blocked attempt
        db.log_activity(current_user["id"], "blocked_ask_question", f"Question: {payload.question} | Reason: {justification}")
        raise HTTPException(status_code=400, detail=justification)
        
    elif status == "CONVERSATIONAL":
        db.log_activity(current_user["id"], "ask_question_conversational", f"Question: {payload.question}")
        
        async def conversational_stream():
            metadata = {
                "type": "metadata",
                "sources": [],
                "sub_queries": [],
                "confidence": "high"
            }
            yield json.dumps(metadata) + "\n"
            yield json.dumps({"type": "token", "content": justification}) + "\n"
            
        return StreamingResponse(
            conversational_stream(),
            media_type="application/x-ndjson"
        )

    # 2. Proceed with RAG
    engine = get_answer_engine()
    
    # Log the question being asked by this user
    db.log_activity(current_user["id"], "ask_question", f"Question: {payload.question} | County: {payload.county}")
    
    return StreamingResponse(
        engine.answer_agent_stream(
            question=payload.question,
            county=payload.county,
            top_k=payload.top_k
        ),
        media_type="application/x-ndjson"
    )

@app.post("/api/search")
@limiter.limit("30/minute")
async def search_chunks(request: Request, payload: SearchRequest, current_user: dict = Depends(get_current_user)):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")

    retrieval_mode = "semantic"
    warning = None
    try:
        retriever = get_retriever()
        chunks = retriever.retrieve(
            query=query,
            top_k=max(1, min(payload.top_k, 20)),
            county_filter=payload.county,
        )
    except Exception as e:
        logger.error("Semantic search failed: %s", e)
        retrieval_mode = "lexical_fallback"
        warning = f"Semantic embeddings unavailable; showing keyword-ranked excerpts instead. {e}"
        chunks = lexical_search_chunks(query, payload.county, max(1, min(payload.top_k, 20)))

    db.log_activity(current_user["id"], "semantic_search", f"Query: {query} | County: {payload.county}")
    if retrieval_mode == "semantic":
        results = [chunk.to_dict() for chunk in chunks]
    else:
        results = chunks
    return {"results": results, "retrieval_mode": retrieval_mode, "warning": warning}

@app.post("/api/topics/analyze")
@limiter.limit("20/minute")
async def analyze_topics(request: Request, payload: TopicAnalyzeRequest, current_user: dict = Depends(get_current_user)):
    if not payload.county or payload.county == "all":
        raise HTTPException(status_code=400, detail="Select a county before running topic analysis.")

    top_k = max(1, min(payload.top_k, 10))
    results = []

    try:
        retriever = get_retriever()
        for topic_key, topic_info in TOPICS.items():
            chunks = retriever.retrieve(
                topic_info["query"],
                top_k=top_k,
                county_filter=payload.county,
            )
            best_distance = chunks[0].distance if chunks else None
            results.append({
                "topic_key": topic_key,
                "topic": topic_info["display_name"],
                "description": topic_info["description"],
                "has_evidence": bool(chunks and best_distance is not None and best_distance < 0.6),
                "best_distance": best_distance,
                "sources": [chunk.to_dict() for chunk in chunks],
            })
    except Exception as e:
        logger.error("Topic analysis failed: %s", e)
        results = []
        for topic_key, topic_info in TOPICS.items():
            query = " ".join([topic_info["query"]] + topic_info["keywords"])
            sources = lexical_search_chunks(query, payload.county, top_k)
            best_distance = sources[0]["distance"] if sources else None
            results.append({
                "topic_key": topic_key,
                "topic": topic_info["display_name"],
                "description": topic_info["description"],
                "has_evidence": bool(sources),
                "best_distance": best_distance,
                "sources": sources,
                "retrieval_mode": "lexical_fallback",
            })

    db.log_activity(current_user["id"], "topic_analysis", f"County: {payload.county}")
    return {"county": payload.county, "results": results}

@app.post("/api/feedback")
@limiter.limit("60/minute")
async def submit_feedback(request: Request, payload: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    db.log_feedback(
        current_user["id"],
        payload.question,
        payload.answer,
        payload.rating,
        json.dumps(payload.sources),
    )
    db.log_activity(current_user["id"], "answer_feedback", f"Rating: {payload.rating}")
    return {"message": "Feedback saved."}

@app.get("/api/counties")
async def get_counties():
    return {"counties": get_canonical_counties()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
