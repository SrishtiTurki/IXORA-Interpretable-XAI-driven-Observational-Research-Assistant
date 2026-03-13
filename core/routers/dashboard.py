# routers/dashboard.py

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()   # ← this line is required!

@router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    path = Path("frontend/dashboard.html")
    if not path.is_file():
        return "<h1>dashboard.html not found</h1>"
    return path.read_text(encoding="utf-8")