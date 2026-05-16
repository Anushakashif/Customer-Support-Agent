import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# ── Database URL ──────────────────────────────────────────────────────────────
# SQLite for local dev — change to PostgreSQL URL in production:
# "postgresql://user:password@localhost:5432/support_db"
DATABASE_URL = "sqlite:///./support_tickets.db"

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite with FastAPI
    poolclass=StaticPool,                        # single connection pool for SQLite
    echo=False                                   # set True to see raw SQL in logs
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

# ── Base class for all models ─────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ── Dependency for FastAPI routes ─────────────────────────────────────────────
def get_db():
    """
    Yields a database session for each request.
    Automatically closes it when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Create all tables ─────────────────────────────────────────────────────────
def init_db():
    """
    Creates all tables defined in models.py.
    Called once at application startup.
    """
    from db.models import Ticket, ConversationTurn   # import here to avoid circular imports
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")