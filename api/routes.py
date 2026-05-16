import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from graph.workflow import run_support_pipeline
from graph.state import TicketStatus
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import log_ticket

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Request / Response models ─────────────────────────────────────────────────
class MessageRequest(BaseModel):
    customer_id:          str
    message:              str
    conversation_history: Optional[List[dict]] = []

class MessageResponse(BaseModel):
    customer_id:       str
    response:          str
    ticket_status:     str
    category:          Optional[str] = None
    priority:          Optional[str] = None
    confidence_score:  Optional[float] = None
    should_escalate:   bool = False
    escalation_reason: Optional[str] = None

# ── Process message endpoint ──────────────────────────────────────────────────
@router.post("/message", response_model=MessageResponse)
async def process_message(
    request: MessageRequest,
    db: Session = Depends(get_db)       # ← inject db session
):
    logger.info(f"Received message from customer: {request.customer_id}")

    try:
        result = run_support_pipeline(
            customer_id=          request.customer_id,
            customer_message=     request.message,
            conversation_history= request.conversation_history
        )

        # Log ticket to database
        ticket = log_ticket(db, result)
        logger.info(f"Ticket #{ticket.id} logged to database")

        return MessageResponse(
            customer_id=      request.customer_id,
            response=         result["final_response"],
            ticket_status=    result["ticket_status"].value,
            category=         result["category"].value if result["category"] else None,
            priority=         result["priority"].value if result["priority"] else None,
            confidence_score= result["confidence_score"],
            should_escalate=  result["should_escalate"],
            escalation_reason=result.get("escalation_reason"),
        )

    except Exception as e:
        logger.error(f"Pipeline error for customer {request.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check endpoint ─────────────────────────────────────────────────────
@router.get("/health")
async def health():
    return {"status": "healthy"}


# ── Test endpoint (simulate a message without full pipeline) ──────────────────
@router.get("/test")
async def test():
    return {
        "message": "Multi-Agent Support API is running",
        "agents":  ["triage", "answer", "escalation"],
        "status":  "ready"
    }