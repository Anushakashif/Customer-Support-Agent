import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from langgraph.graph import StateGraph, END
from graph.state import SupportState, TicketStatus
from agents.triage_agent import triage_node
from agents.answer_agent import answer_node
from agents.escalation_agent import escalation_node

logger = logging.getLogger(__name__)

# ── Routing function ──────────────────────────────────────────────────────────
def route_after_answer(state: SupportState) -> str:
    """
    Called after Answer Agent runs.
    Decides whether to escalate or end the conversation.
    """
    if state.get("should_escalate"):
        logger.info("Routing → Escalation Agent")
        return "escalate"
    else:
        logger.info("Routing → END (resolved)")
        return "end"


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph workflow.
    Returns a compiled graph ready to run.
    """

    # Initialize graph with our shared state
    graph = StateGraph(SupportState)

    # ── Add nodes (one per agent) ─────────────────────────────────────────────
    graph.add_node("triage",    triage_node)
    graph.add_node("answer",    answer_node)
    graph.add_node("escalate",  escalation_node)

    # ── Define edges (flow between agents) ────────────────────────────────────

    # Always start with triage
    graph.set_entry_point("triage")

    # Triage always goes to answer
    graph.add_edge("triage", "answer")

    # After answer → check confidence → route accordingly
    graph.add_conditional_edges(
        "answer",               # from this node
        route_after_answer,     # call this function to decide
        {
            "escalate": "escalate",   # if returns "escalate" → go to escalation node
            "end":      END           # if returns "end" → finish
        }
    )

    # Escalation is always the final step
    graph.add_edge("escalate", END)

    # Compile and return
    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# ── Singleton graph instance ──────────────────────────────────────────────────
# Built once, reused across all requests
support_graph = build_graph()


# ── Main run function ─────────────────────────────────────────────────────────
def run_support_pipeline(
    customer_id: str,
    customer_message: str,
    conversation_history: list = []
) -> SupportState:
    """
    Entry point to run a message through the full pipeline.
    Returns the final state with all agent outputs.
    """
    logger.info(f"Pipeline started for customer: {customer_id}")

    # Build initial state
    initial_state: SupportState = {
        "customer_id":          customer_id,
        "customer_message":     customer_message,
        "conversation_history": conversation_history,

        # These will be filled by agents
        "category":             None,
        "priority":             None,
        "extracted_entities":   {},
        "generated_answer":     None,
        "confidence_score":     None,
        "retrieved_docs":       [],
        "should_escalate":      False,
        "escalation_reason":    None,
        "escalation_summary":   None,
        "final_response":       None,
        "ticket_status":        TicketStatus.OPEN,
    }

    # Run the graph
    final_state = support_graph.invoke(initial_state)

    logger.info(f"Pipeline complete → Status: {final_state['ticket_status']}")
    return final_state