# routers/public.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter(tags=["public", "frontend"])

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_home():
    path = Path("auth/index.html")
    if not path.is_file():
        return "<h1>Error: index.html not found</h1>"
    return path.read_text(encoding="utf-8")


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def serve_login():
    path = Path("auth/login.html")
    if not path.is_file():
        return "<h1>Error: login.html not found</h1>"
    return path.read_text(encoding="utf-8")