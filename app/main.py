"""Main FastAPI application for the code evaluation system."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from app.database import init_db
from app.evaluator import validate_config
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown events.

    Initializes the database and validates configuration on startup.
    """
    logger.info("Starting LLM Code Evaluator...")

    # Validate configuration
    try:
        config = validate_config()
        logger.info("Configuration validated. Provider: %s", config["llm_provider"])
    except Exception as e:
        logger.error("Configuration validation failed: %s", str(e))

    # Initialize database
    init_db()
    logger.info("Application started successfully")

    yield

    logger.info("Shutting down LLM Code Evaluator...")


app = FastAPI(
    title="LLM Code Evaluator",
    description="A Python code quality evaluation system powered by LLMs",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api", tags=["evaluation"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request) -> HTMLResponse:
    """Serve the main HTML form for code submission.

    Args:
        request: The incoming request.

    Returns:
        Rendered HTML template.
    """
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Status dictionary.
    """
    return {"status": "healthy", "service": "llm-code-evaluator"}
