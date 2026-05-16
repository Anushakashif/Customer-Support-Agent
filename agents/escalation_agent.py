import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.state import SupportState, TicketStatus
from config.settings import settings
from integrations.slack_notifier import send_escalation_notification

logger = logging.getLogger(__name__)

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,          # deterministic — summaries should be consistent
    api_key=settings.OPENAI_API_KEY
)

# ── Prompt ───────────────────────────────────────────────────────────────────
escalation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a support escalation agent. Your job is to prepare a clear,
        structured handoff summary for a human support agent.

        The AI could not confidently answer this customer's question.
        Summarize the situation so a human can resolve it quickly.

        Return ONLY a JSON object with no extra text:
        {{
            "escalation_reason": "<one sentence: why the AI could not resolve this>",
            "summary": "<2-3 sentences: what the customer needs, what was tried, what the human should do>",
            "suggested_action": "<one concrete action the human agent should take>",
            "customer_facing_message": "<polite message to send the customer explaining they are being connected to a human>"
        }}"""
    ),
    (
        "human",
        """Customer message: {customer_message}

        Category: {category}
        Priority: {priority}
        Confidence score: {confidence_score}

        AI's attempted answer: {generated_answer}

        Conversation history: {conversation_history}

        Prepare the escalation handoff."""
    )
])

# ── Escalation chain ──────────────────────────────────────────────────────────
escalation_chain = escalation_prompt | llm

# ── Main node function ────────────────────────────────────────────────────────
def escalation_node(state: SupportState) -> SupportState:
    """
    Generates an escalation summary and customer-facing message.
    Marks the ticket as escalated.
    """
    logger.info(f"Escalation Agent started for customer: {state['customer_id']}")

    try:
        # Format conversation history for the prompt
        history = state.get("conversation_history", [])
        history_text = "\n".join([
            f"{turn['role'].upper()}: {turn['content']}"
            for turn in history
        ]) if history else "No prior conversation."

        # Call the LLM
        response = escalation_chain.invoke({
            "customer_message":   state["customer_message"],
            "category":           str(state["category"].value),
            "priority":           str(state["priority"].value),
            "confidence_score":   state.get("confidence_score", 0.0),
            "generated_answer":   state.get("generated_answer", "No answer attempted."),
            "conversation_history": history_text
        })

        # Parse response
        raw = response.content.strip()
        result = json.loads(raw)

        escalation_reason   = result.get("escalation_reason", "Low confidence score.")
        summary             = result.get("summary", "")
        suggested_action    = result.get("suggested_action", "")
        customer_message    = result.get("customer_facing_message", (
            "I'm connecting you with a human support agent who will "
            "be able to assist you further. Please hold on."
        ))

        logger.info(f"Escalation reason: {escalation_reason}")
        logger.info(f"Suggested action: {suggested_action}")

        # Notify human agents on Slack
        send_escalation_notification({
            **state,
            "escalation_reason":  escalation_reason,
            "escalation_summary": summary,
        })

        return {
            **state,
            "escalation_reason":  escalation_reason,
            "escalation_summary": summary,
            "final_response":     customer_message,
            "ticket_status":      TicketStatus.ESCALATED,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Escalation Agent JSON parse error: {e}")
        return {
            **state,
            "escalation_reason":  "System error during escalation.",
            "escalation_summary": "Unable to generate summary.",
            "final_response":     "We're connecting you with a human agent shortly.",
            "ticket_status":      TicketStatus.ESCALATED,
        }

    except Exception as e:
        logger.error(f"Escalation Agent unexpected error: {e}")
        raise