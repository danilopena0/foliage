"""Main FastAPI application for the code evaluation system."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load environment variables from .env.example file
load_dotenv()

from app.database import init_db
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler for startup/shutdown events.

    Initializes the database on startup.
    """
    init_db()
    yield


app = FastAPI(
    title="LLM Code Evaluator",
    description="A Python code quality evaluation system powered by Claude AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api", tags=["evaluation"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request) -> HTMLResponse:
    """
    Serve the main HTML form for code submission.

    Args:
        request: The incoming request.

    Returns:
        Rendered HTML template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        Status dictionary.
    """
    return {"status": "healthy", "service": "llm-code-evaluator"}
