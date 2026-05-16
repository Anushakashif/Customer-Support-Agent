from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Boolean,
    DateTime, Text, Integer, ForeignKey
)
from sqlalchemy.orm import relationship
from db.database import Base


# ── Ticket table ──────────────────────────────────────────────────────────────
class Ticket(Base):
    __tablename__ = "tickets"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    customer_id       = Column(String(100), nullable=False, index=True)
    customer_message  = Column(Text, nullable=False)

    # Triage Agent output
    category          = Column(String(50), nullable=True)
    priority          = Column(String(50), nullable=True)

    # Answer Agent output
    generated_answer  = Column(Text, nullable=True)
    confidence_score  = Column(Float, nullable=True)

    # Escalation
    should_escalate   = Column(Boolean, default=False)
    escalation_reason = Column(Text, nullable=True)
    escalation_summary= Column(Text, nullable=True)

    # Final
    final_response    = Column(Text, nullable=True)
    ticket_status     = Column(String(50), default="open")

    # Timestamps
    created_at        = Column(DateTime, default=datetime.utcnow)
    resolved_at       = Column(DateTime, nullable=True)

    # Relationship to conversation turns
    turns             = relationship("ConversationTurn", back_populates="ticket")

    def __repr__(self):
        return f"<Ticket id={self.id} customer={self.customer_id} status={self.ticket_status}>"


# ── Conversation turn table ───────────────────────────────────────────────────
class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id  = Column(Integer, ForeignKey("tickets.id"), nullable=False)
    role       = Column(String(20), nullable=False)   # "customer" or "agent"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to ticket
    ticket     = relationship("Ticket", back_populates="turns")

    def __repr__(self):
        return f"<Turn id={self.id} role={self.role} ticket={self.ticket_id}>"


from sqlalchemy.orm import Session
from graph.state import SupportState

def log_ticket(db: Session, state: SupportState) -> Ticket:
    """
    Takes the final pipeline state and saves it
    as a ticket record in the database.
    """
    ticket = Ticket(
        customer_id=       state["customer_id"],
        customer_message=  state["customer_message"],
        category=          state["category"].value if state["category"] else None,
        priority=          state["priority"].value if state["priority"] else None,
        generated_answer=  state.get("generated_answer"),
        confidence_score=  state.get("confidence_score"),
        should_escalate=   state.get("should_escalate", False),
        escalation_reason= state.get("escalation_reason"),
        escalation_summary=state.get("escalation_summary"),
        final_response=    state.get("final_response"),
        ticket_status=     state["ticket_status"].value,
        resolved_at=       datetime.utcnow() if state["ticket_status"].value == "resolved" else None
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket