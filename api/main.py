import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.settings import settings
from api.routes import router
from integrations.twilio_webhook import router as twilio_router
from rag.embedder import get_vector_store
from db.database import init_db

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),                  # print to console
        logging.FileHandler("logs/app.log", encoding="utf-8")  # save to file
    ]
)

logger = logging.getLogger(__name__)

# ── Lifespan (runs on startup and shutdown) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME}...")
    init_db()
    get_vector_store()      # preload FAISS index into memory
    logger.info("FAISS index loaded and ready")
    logger.info("All agents initialized and ready")
    yield
    # Shutdown
    logger.info("Shutting down...")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-Agent Customer Support System using LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows frontend or Twilio webhooks to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")
app.include_router(twilio_router, prefix="/webhook")

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app":    settings.APP_NAME,
        "status": "running",
        "version": "1.0.0"
    }