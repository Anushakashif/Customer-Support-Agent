from typing import TypedDict, Optional, List
from enum import Enum

# ── Categories a message can belong to ──────────────────────────────────────
class Category(str, Enum):
    BILLING     = "billing"
    TECHNICAL   = "technical"
    COMPLAINT   = "complaint"
    GENERAL     = "general"
    UNKNOWN     = "unknown"

# ── Priority levels assigned by Triage Agent ────────────────────────────────
class Priority(str, Enum):
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    URGENT  = "urgent"

# ── Final status of the ticket ───────────────────────────────────────────────
class TicketStatus(str, Enum):
    OPEN        = "open"
    RESOLVED    = "resolved"
    ESCALATED   = "escalated"

# ── The main state object passed between all agents ──────────────────────────
class SupportState(TypedDict):

    # Customer info
    customer_id:        str
    customer_message:   str
    conversation_history: List[dict]

    # Triage Agent fills these
    category:           Optional[Category]
    priority:           Optional[Priority]
    extracted_entities: Optional[dict]       # e.g. {"order_id": "123", "account": "xyz"}

    # Answer Agent fills these
    generated_answer:   Optional[str]
    confidence_score:   Optional[float]
    retrieved_docs:     Optional[List[str]]  # chunks pulled from knowledge base

    # Escalation Agent fills these
    should_escalate:    bool
    escalation_reason:  Optional[str]
    escalation_summary: Optional[str]

    # Final output
    final_response:     Optional[str]
    ticket_status:      TicketStatus